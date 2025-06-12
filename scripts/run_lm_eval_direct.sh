#!/bin/bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=a100:4
#SBATCH --mem=64000M
#SBATCH --time=1-00:00
#SBATCH --account=rrg-lilimou
#SBATCH --output=slurm-logs/%j-llama-3-70B-distributed.out


nvidia-smi
nvidia-smi topo -m

export HEAD_NODE=$(hostname) # store head node's address
export HEAD_NODE_PORT=34568 # choose a port on the main node to start accelerate's main process


# export CUDA_VISIBLE_DEVICES=2
# llmc=/home/zonglin/Documents/llmc
# lm_eval=./llmc/lm-evaluation-harness
# export PYTHONPATH=$llmc:$PYTHONPATH
# export PYTHONPATH=$llmc:$lm_eval:$PYTHONPATH
# Replace the config file (i.e., RTN with algorithm-transformed model path or notate quant with original model path) 
# with the one you want to use. `--quarot` is depend on the transformation algorithm used before.

accelerate launch \
    --multi_gpu \
    --gpu_ids="all" \
    --num_machines=$SLURM_NNODES \
    --machine_rank=$SLURM_NODEID \
    --num_processes=8 \
    --main_process_ip=$HEAD_NODE \
    --main_process_port=$HEAD_NODE_PORT \
    --config_file accl-config/2node-fsdp.yaml \
    -m lm_eval \
        --model hf \
        --model_args pretrained="meta-llama/meta-Llama-3-70B,parallelize=True,max_length=2048,device_map=cuda" \
        --tasks wikitext \
        --batch_size 1 \
        --output_path runs/lm_eval \
        --log_samples \
        --limit 128 \