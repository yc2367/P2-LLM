import torch.nn.functional as F
from torch import nn

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from .quant_config import QuantConfig
from .quantizer import kv_quant_function

import logging
logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class QuantLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, q_config: QuantConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # KV-cache quantization config
        self.q_config = q_config
        self.kv_quant_method = q_config.kv_quant_method
        self.k_bits = q_config.k_bits
        self.v_bits = q_config.v_bits
        self.k_group_size = q_config.k_group_size
        self.v_group_size = q_config.v_group_size
        self.kv_quant_before_attn = q_config.kv_quant_before_attn
        self.kv_residual_len = q_config.kv_residual_len
        self.apply_k_bias = q_config.apply_k_bias
        self.apply_k_scale = q_config.apply_k_scale
        self.k_bias = None
        self.k_scale = None

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()
        if q_len > 1:
            is_prefill = True
        else:
            is_prefill = False

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Update sequence length
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[-1]
        
        # Rotary Positional Embedding (RoPE)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        #NOTE: Added by (Yuzong Chen)
        if self.kv_quant_before_attn:
            if past_key_value is not None: # Decoding Stage
                key_states_quant = past_key_value[0] # quantized key
                key_states_float = past_key_value[1] # full-precision residual key
                value_states_quant = past_key_value[2] # quantized value
                value_states_float = past_key_value[3] # full-precision residual value

                if key_states_float is not None:
                    key_states_float = torch.cat([key_states_float, key_states], dim=2)
                else:
                    key_states_float = key_states
                if value_states_float is not None:
                    value_states_float = torch.cat([value_states_float, value_states], dim=2)
                else:
                    value_states_float = value_states

                if key_states_float.shape[-2] == self.kv_residual_len:
                    key_states_quant_new, value_states_quant_new = kv_quant_function(
                        key_states_float, value_states_float,
                        self.quant_config, k_bias=self.k_bias, k_scale=self.k_scale
                    )
                    key_states_float = None
                    value_states_float = None
                    if key_states_quant is not None:
                        key_states_quant = torch.cat([key_states_quant, key_states_quant_new], dim=2)
                    else:
                        key_states_quant = key_states_quant_new
                    if value_states_quant is not None:
                        value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    else:
                        value_states_quant = value_states_quant_new
            else: # Prefill Stage
                # Calculate scale and bias for key-smoothing
                self.k_scale = key_states.abs().amax(dim=2, keepdim=True).pow(0.6)
                self.k_bias = key_states.mean(dim=2, keepdim=True)

                kv_states_float_len = key_states.shape[-2] % self.kv_residual_len
                if kv_states_float_len != 0:
                    if key_states.shape[-2] < self.kv_residual_len:
                        key_states_quant = None
                        key_states_float = key_states
                        value_states_quant = None
                        value_states_float = value_states
                    else:
                        key_states_quant = key_states[:, :, :-kv_states_float_len, :].contiguous()
                        key_states_float = key_states[:, :, -kv_states_float_len:, :].contiguous()
                        value_states_quant = value_states[:, :, :-kv_states_float_len, :].contiguous()
                        value_states_float = value_states[:, :, -kv_states_float_len:, :].contiguous()
                else:
                    key_states_quant = key_states
                    key_states_float = None
                    value_states_quant = value_states
                    value_states_float = None
                    
                key_states_quant, value_states_quant = kv_quant_function(
                    key_states_quant, value_states_quant,
                    self.quant_config, k_bias=self.k_bias, k_scale=self.k_scale
                )
            
            if key_states_quant is None:
                key_states_full = key_states_float
            elif key_states_float is None:
                key_states_full = key_states_quant
            else:
                key_states_full = torch.cat([key_states_quant, key_states_float], dim=2)

            if value_states_quant is None:
                value_states_full = value_states_float
            elif value_states_float is None:
                value_states_full = value_states_quant
            else:
                value_states_full = torch.cat([value_states_quant, value_states_float], dim=2)
            
            # For GQA, repeat k/v heads if n_kv_heads < n_heads
            key_states_full = repeat_kv(key_states_full, self.num_key_value_groups)
            value_states_full = repeat_kv(value_states_full, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states_full.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
                
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states_full) 
        else:
            if past_key_value is not None: # Decoding Stage
                key_states_quant = past_key_value[0] # quantized key
                key_states_float = past_key_value[1] # full-precision residual key
                value_states_quant = past_key_value[2] # quantized value
                value_states_float = past_key_value[3] # full-precision residual value

                if key_states_float is not None:
                    key_states_float = torch.cat([key_states_float, key_states], dim=2)
                else:
                    key_states_float = key_states
                if value_states_float is not None:
                    value_states_float = torch.cat([value_states_float, value_states], dim=2)
                else:
                    value_states_float = value_states
            else:
                self.k_scale = key_states.abs().amax(dim=2, keepdim=True).pow(0.6)
                self.k_bias = key_states.mean(dim=2, keepdim=True)

                kv_states_float_len = key_states.shape[-2] % self.kv_residual_len
                if kv_states_float_len != 0:
                    if key_states.shape[-2] < self.kv_residual_len:
                        key_states_quant = None
                        key_states_float = key_states
                        value_states_quant = None
                        value_states_float = value_states
                    else:
                        key_states_quant = key_states[:, :, :-kv_states_float_len, :].contiguous()
                        key_states_float = key_states[:, :, -kv_states_float_len:, :].contiguous()
                        value_states_quant = value_states[:, :, :-kv_states_float_len, :].contiguous()
                        value_states_float = value_states[:, :, -kv_states_float_len:, :].contiguous()
                else:
                    key_states_quant = key_states
                    key_states_float = None
                    value_states_quant = value_states
                    value_states_float = None

            if key_states_quant is None:
                key_state_full = key_states_float
            elif key_states_float is None:
                key_state_full = key_states_quant
            else:
                key_state_full = torch.cat([key_states_quant, key_states_float], dim=2)

            if value_states_quant is None:
                value_states_full = value_states_float
            elif value_states_float is None:
                value_states_full = value_states_quant
            else:
                value_states_full = torch.cat([value_states_quant, value_states_float], dim=2)

            attn_weights = torch.matmul(query_states, key_state_full.transpose(2, 3)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )
            
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states_full) 

            if past_key_value is not None: # Decoding Stage
                if key_states_float.shape[-2] == self.kv_residual_len:
                    key_states_quant_new, value_states_quant_new = kv_quant_function(
                        key_states_float, value_states_float,
                        self.quant_config, k_bias=self.k_bias, k_scale=self.k_scale
                    )
                    key_states_float = None
                    value_states_float = None
                    if key_states_quant is not None:
                        key_states_quant = torch.cat([key_states_quant, key_states_quant_new], dim=2)
                    else:
                        key_states_quant = key_states_quant_new
                    if value_states_quant is not None:
                        value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    else:
                        value_states_quant = value_states_quant_new
            else:
                kv_states_float_len = key_states.shape[-2] % self.kv_residual_len
                if kv_states_float_len != 0:
                    if key_states.shape[-2] < self.kv_residual_len:
                        key_states_quant = None
                        key_states_float = key_states
                        value_states_quant = None
                        value_states_float = value_states
                    else:
                        key_states_quant = key_states[:, :, :-kv_states_float_len, :].contiguous()
                        key_states_float = key_states[:, :, -kv_states_float_len:, :].contiguous()
                        value_states_quant = value_states[:, :, :-kv_states_float_len, :].contiguous()
                        value_states_float = value_states[:, :, -kv_states_float_len:, :].contiguous()
                else:
                    key_states_quant = key_states
                    key_states_float = None
                    value_states_quant = value_states
                    value_states_float = None
                    
                key_states_quant, value_states_quant = kv_quant_function(
                    key_states_quant, value_states_quant,
                    self.quant_config, k_bias=self.k_bias, k_scale=self.k_scale
                )

        past_key_value = (key_states_quant, key_states_float, value_states_quant, value_states_float, kv_seq_len) if use_cache else None
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, quant_config: QuantConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = QuantLlamaAttention(config=config, quant_config=quant_config)
        self.mlp = LlamaMLP(config=config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
            

class QuantLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, quant_config: QuantConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, quant_config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class QuantLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, quant_config: QuantConfig):
        super().__init__(config)
        self.model = QuantLlamaModel(config, quant_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # if past_key_values is not None:
        #     past_length = past_key_values[0][-1]
        #     # Some generation methods already pass only the last input ID
        #     if input_ids.shape[1] > past_length:
        #         remove_prefix_length = past_length
        #     else:
        #         # Default to old behavior: keep only final ID
        #         remove_prefix_length = input_ids.shape[1] - 1

        #     input_ids = input_ids[:, remove_prefix_length:]
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past



class ExperimentalLlamaAttention(LlamaAttention):
    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int] = None, compress_config=None
    ):
        super().__init__(config, layer_idx)
        self.compress_config = compress_config

class ExperimentalLlamaSdpaAttention(ExperimentalLlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if q_len > 1:
            prefill = True
        else:
            prefill = False
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            if self.compress_config is not None:
                if self.compress_config.streaming[self.layer_idx] is True:
                    if len(past_key_value) >= self.layer_idx + 1:
                        past_key, past_value = past_key_value[self.layer_idx]
                    else:
                        past_key, past_value = key_states, value_states
                    
                    _, _, seq_len, _ = past_key.shape
                    if prefill or seq_len % self.compress_config.streaming_gap[self.layer_idx] == 0:
                        past_key, past_value = self._apply_kv_compress(past_key, past_value)

                if not prefill:
                    #NOTE(brian1009): Workaround for the missing interface for editing KV-Cache
                    assert self.layer_idx < len(past_key_value.key_cache), "Past key cache is not initialized correctly"
                    assert self.layer_idx < len(past_key_value.value_cache), "Past value cache is not initialized correctly"
                    past_key_value.key_cache[self.layer_idx] = past_key
                    past_key_value.value_cache[self.layer_idx] = past_value
                prefill = False        

            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    
    def _apply_kv_compress(self, past_key, past_value):
        if self.compress_config.compress_method[self.layer_idx] == "KIVI":
            bsz, num_heads, seq_len, head_dim = past_key.shape
            fixed_length = seq_len // self.compress_config.group_size[
                self.layer_idx
            ] * self.compress_config.group_size[self.layer_idx]
            residual_key = past_key[:, :, fixed_length:, :]
            residual_value = past_value[:, :, fixed_length:, :]
            past_key, past_value = past_key[:,:,:fixed_length,:], past_value[:,:,:fixed_length,:]

        (past_key, past_value) = compress_insert_function(
            previous_key=past_key,
            previous_value=past_value,
            compress_config=self.compress_config,
            layer_idx=self.layer_idx,
        )
        
        if self.compress_config.compress_method[self.layer_idx] == "KIVI":
            past_key = torch.cat([past_key, residual_key], dim=2)
            past_value = torch.cat([past_value, residual_value], dim=2)
            
        return past_key, past_value




def convert_attention_to_experimental(model, compress_config):
    producer_layer = None
    layer_counter = {'idx': 0}
    assert model.config._attn_implementation == 'sdpa', 'Only support SDPA implementation'
    def recurse_convert(parent_module):
        nonlocal producer_layer
        for name, module in parent_module._modules.items():
            if len(list(module.children())) > 0:
                recurse_convert(module)
            if isinstance(module, LlamaAttention):
                device = next(module.parameters()).device
                dtype = next(module.parameters()).dtype
                new_module = ExperimentalLlamaSdpaAttention(
                    compress_config=compress_config, 
                    config=model.config,
                    layer_idx=layer_counter['idx']
                ).to(dtype).to(device)
                new_module.load_state_dict(module.state_dict(), strict=False)
                parent_module._modules[name] = new_module
                layer_counter['idx'] += 1
    
    recurse_convert(model)
    
    return model