#!/bin/bash
#SBATCH --nodes=1
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


export CUDA_VISIBLE_DEVICES=2,3
export NUM_PROCESSES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

python -m lm_eval \
        --model hf \
        --model_args pretrained="runs/opt-13b_rtn_w4/fake_quant_model",parallelize=True,dtype=float16,trust_remote_code=true \
        --tasks wikitext \
        --batch_size 1 \
        --output_path runs/lm_eval \
        # --log_samples \