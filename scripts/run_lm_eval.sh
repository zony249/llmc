export CUDA_VISIBLE_DEVICES=2,3
llmc=/home/zonglin/Documents/llmc
lm_eval=./llmc/lm-evaluation-harness
# export PYTHONPATH=$llmc:$PYTHONPATH
# export PYTHONPATH=$llmc:$lm_eval:$PYTHONPATH
# Replace the config file (i.e., RTN with algorithm-transformed model path or notate quant with original model path) 
# with the one you want to use. `--quarot` is depend on the transformation algorithm used before.
accelerate launch --multi_gpu --num_processes 2 ${llmc}/tools/llm_eval.py \
    --config ${llmc}/configs/custom/awq_w_a.yml \
    --model hf \
    --quarot \
    --tasks wikitext \
    --model_args parallelize=True,trust_remote_code=True \
    --batch_size 1 \
    --output_path runs/lm_eval \
    --log_samples