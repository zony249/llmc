#!/bin/bash 



export MODEL=runs/vllm/fake_quant_model
export MODEL_PATH=$MODEL

export CUDA_VISIBLE_DEVICES=2,3
# export PYTHONPATH=$(pwd)/llm-awq 
# echo $PYTHONPATH

python -m awq.entry --model_path $MODEL_PATH\
    --tasks wikitext \