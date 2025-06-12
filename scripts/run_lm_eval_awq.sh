#!/bin/bash 



export MODEL=/home/zonglin/large-file-storage/meta-llama/Llama-2-7b-hf
export MODEL_PATH=$MODEL

export CUDA_VISIBLE_DEVICES=3
# export PYTHONPATH=$(pwd)/llm-awq 
# echo $PYTHONPATH

# python -m awq.entry --model_path $MODEL_PATH\
#     --tasks wikitext \
export NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)


accelerate launch \
    --num_processes $NUM_PROCESSES \
    --num_machines 1 \
    evaluation/entry.py \
        --model_path $MODEL_PATH\
        --tasks wikitext \

# python llm-awq/entry.py \
#     --model_path $MODEL_PATH\
#     --tasks wikitext \