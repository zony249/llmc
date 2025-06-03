#!/bin/bash

export CUDA_VISIBLE_DEVICES=2
# llmc=/home/zonglin/Documents/llmc
# lm_eval=./llmc/lm-evaluation-harness
# export PYTHONPATH=$llmc:$PYTHONPATH
# export PYTHONPATH=$llmc:$lm_eval:$PYTHONPATH
# Replace the config file (i.e., RTN with algorithm-transformed model path or notate quant with original model path) 
# with the one you want to use. `--quarot` is depend on the transformation algorithm used before.
accelerate launch --num_processes 1 -m lm_eval \
    --model hf \
    --model_args pretrained="/home/zonglin/large-file-storage/meta-llama/meta-Llama-3-8B,parallelize=False,max_length=2048,trust_remote_code=True" \
    --tasks wikitext \
    --batch_size 1 \
    --output_path runs/lm_eval \
    --log_samples \
    --limit 128 \