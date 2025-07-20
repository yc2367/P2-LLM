# ========== QoQ with Post-Scale Zero Point ===============================

# ========== QoQ (W4A8KV4 with progressive weight quantization) ==========

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-7B
model_name="llama-7b"
model_path="huggyllama/llama-7b"
CUDA_VISIBLE_DEVICES=0,1 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.3 --smooth-proj-beta 0.7 \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-13B
model_name="llama-13b"
model_path="huggyllama/llama-13b"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.2 --smooth-proj-beta 0.8 \
    --smooth-attn-strategy GridSearch --smooth-attn-beta " -2" \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-2-7B
model_name="llama-2-7b"
model_path="meta-llama/Llama-2-7b-hf"
CUDA_VISIBLE_DEVICES=0,1 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.2 --smooth-proj-beta 0.8 \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-2-13B
model_name="llama-2-13b"
model_path="meta-llama/Llama-2-13b-hf"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.35 --smooth-proj-beta 0.65 \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3.1-8B
model_name="llama-3.1-8b"
model_path="meta-llama/Llama-3.1-8B"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.35 --smooth-proj-beta 0.65 \
    --smooth-attn-strategy GridSearch --smooth-attn-beta " -2" \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3.1-8B-Instruct, weight group-size = 128
model_name="llama-3.1-8b-ins"
model_path="meta-llama/Llama-3.1-8B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.3 --smooth-proj-beta 0.7 \
    --smooth-attn-strategy GridSearch --smooth-attn-beta " -2" \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3.1-8B-Instruct, weight group-size = 64
model_name="llama-3.1-8b-ins"
model_path="meta-llama/Llama-3.1-8B-Instruct"
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g64.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.4 --smooth-proj-beta 0.6 \
    --smooth-attn-strategy GridSearch --smooth-attn-beta " -2" \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3.2-3B
model_name="llama-3.2-3b"
model_path="meta-llama/Llama-3.2-3B"
CUDA_VISIBLE_DEVICES=0 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.35 --smooth-proj-beta 0.65 \
    --smooth-attn-strategy GridSearch --smooth-attn-beta " -2" \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3.2-3B-Instruct, weight group-size = 128
model_name="llama-3.2-3b-ins"
model_path="meta-llama/Llama-3.2-3B-Instruct"
CUDA_VISIBLE_DEVICES=0 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.2 --smooth-proj-beta 0.8 \
    --smooth-attn-strategy GridSearch --smooth-attn-beta " -2" \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Llama-3.2-3B-Instruct, weight group-size = 64
model_name="llama-3.2-3b-ins"
model_path="meta-llama/Llama-3.2-3B-Instruct"
CUDA_VISIBLE_DEVICES=0 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g64.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.3 --smooth-proj-beta 0.7 \
    --smooth-attn-strategy GridSearch --smooth-attn-beta " -2" \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Mistral-7B-v0.3
model_name="mistral-7b-v0.1"
model_path="mistralai/Mistral-7B-v0.1"
CUDA_VISIBLE_DEVICES=0,1 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.15 --smooth-proj-beta 0.85 \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Mistral-7B-v0.3
model_name="mistral-7b-v0.3"
model_path="mistralai/Mistral-7B-v0.3"
CUDA_VISIBLE_DEVICES=0,1 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.25 --smooth-proj-beta 0.75 \
    --save-model "false" \
    #--load-from ""

# QoQ (W4A8KV4 with progressive weight quantization) on Mistral-7B-Instruct-v0.3
model_name="mistral-7b-ins-v0.3"
model_path="mistralai/Mistral-7B-Instruct-v0.3"
CUDA_VISIBLE_DEVICES=0,1 python -m deepcompressor.app.llm.ptq examples/llm/configs/qoq-g128.yaml \
    --model-name ${model_name} --model-path ${model_path} \
    --smooth-proj-alpha 0.05 --smooth-proj-beta 0.95 \
    --save-model "false" \
    #--load-from ""
# ========================================================================

# ========================================================================