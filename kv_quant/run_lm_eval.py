
# Import necessary modules
from utils import (
    load_model_and_tokenizer, 
    add_common_args, 
    add_cache_args, 
    add_compression_args,
    get_compression_config,
    set_seed,
)
import argparse
import torch
import lm_eval
from tqdm import tqdm
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from lm_eval.models.utils import stop_sequences_criteria
import logging
logger = logging.getLogger(__name__)
import os
import json

from BitMod import CompressionConfig

import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

class HuggingFaceCausalLM_withKV(HFLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kv_cache_log = []
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        outputs = self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            **generation_kwargs,
        )
        past_key_values = outputs['past_key_values']
        self.kv_cache_log.append(past_key_values)
        # print(past_key_values[0][1].shape)
        # print(f'# of layers: {len(past_key_values)}')
        # print(f'Size of tuple: {len(past_key_values[0])}')
        # print(f'Shape of KV: {len(past_key_values[0][0].shape)}')
        return outputs['sequences']


def run_lm_eval_zero_shot(
    args, model, tokenizer, 
    batch_size=64, max_length=4096, 
    task_list=["arc_easy", "hellaswag"], 
    limit=None,
    fewshot_as_multiturn=False,
    apply_chat_template=False,
):
    model.seqlen = max_length
    lm_obj = HuggingFaceCausalLM_withKV(pretrained=model, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)

    #lm_obj = BitModLMWrapper(pretrained=model, kv_cache_args=args, tokenizer=tokenizer, add_bos_token=False, batch_size=batch_size)
    # indexes all tasks from the lm_eval/tasks subdirectory.
    # Alternatively, you can set TaskManager(include_path="path/to/my/custom/task/configs")
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting task_manager to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in lm_eval/tasks.
    # simple_evaluate will instantiate its own task_manager is the it is set to None here.
    logger.info(f"Evaluation, Task(s): {task_list}")
    with torch.no_grad():
        results = lm_eval.simple_evaluate( # call simple_evaluate
            model=lm_obj,
            tasks=task_list,
            task_manager=task_manager,
            limit=limit,
            log_samples=True,
            fewshot_as_multiturn=fewshot_as_multiturn,
            apply_chat_template=apply_chat_template,
        ) 
    res = make_table(results)
    
    return results['results']


if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_compression_args(parser)
    #add_cache_args(parser)
    parser.add_argument(
        '--tasks', type=lambda s: [item for item in s.split(',')], default=[],
        help='Task to be evaled'
    )
    parser.add_argument(
        '--batch_size',
        default=8,
        type=int,
        help='batch size for lm_eval tasks'
    )
    parser.add_argument(
        '--limit',
        default=None,
        type=int,
        help='limit number of samples to run'
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print verbose information or not."
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        help="Whether to treat fewshot as multiturn or not."
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Whether to apply chat template or not."
    )
    parser.add_argument(
        "--save_results",
        type=int,
        help="Whether to save the results or not."
    )
    parser.add_argument(
        '--output_dir', type=str, default='results/lm_eval', 
        help='output directory',
    )
    parser.add_argument(
        '--gen_kwargs', type=str, default=None, 
        help='transformer GenerationConfig',
    )
    parser.add_argument(
        "--test_compression",
        action="store_true",
        help="Test compression"
    )
    args = parser.parse_args()  
    
    logger.info(f"Start evaluating with the following configurations:")
    logger.info("Loading model and tokenizer...")
    
    if args.test_compression:
        compress_config = get_compression_config(args)
        logger.info(f"* Bench compression!!!")
        logger.info(f"* Compression method: {args.compress_method}")
        logger.info(f"* Number of bits: {args.quantize_bit}")
        logger.info(f"* Streaming_gap (Residual): {args.streaming_gap}")
        logger.info(f"* Use BitMod: {args.use_bitmod}")
        logger.info(f"* Group Size: {args.group_size}")
    else:
        logger.info(f"* No compression... Bench original LLM model")
        compress_config = None
        
    model, tokenizer = load_model_and_tokenizer(args.model_name_or_path, compression_config=compress_config)
    logger.info("Start running lm_eval zero-shot evaluation...")
    res = run_lm_eval_zero_shot(
        args, model, tokenizer, args.batch_size, 
        task_list=args.tasks, apply_chat_template=args.apply_chat_template, 
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        limit=args.limit,
    )
    
    # Create directory if it doesn't exist
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save results to JSON file
    model_name = args.model_name_or_path.split("/")[-1]
    
    if args.test_compression:
        file_name = f"{model_name}_method_{args.compress_method}_bits-{args.quantize_bit}_gap-{args.streaming_gap}_bitmod-{args.use_bitmod}_group-{args.group_size}"
    else:
        file_name = f"{model_name}_method_fp16"
    
    output_file = os.path.join(output_dir, f"{file_name}.json")
    with open(output_file, "w") as f:
        json.dump(res, f, indent=4)

    print(f"Results saved to {output_file}")
    