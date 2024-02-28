#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --mem=0
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --output slurm_logs/generate-%j.out

module purge 
module load gcc
module load cuda

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -m synthetic_llm_data.src.data_generation.data_generate \
    --name_or_path "" \
    --seed 1 \
    --max_batch_size 16 \
    --max_prompts 200 \
    --use_beam_search True \
    --system_prompt "" \
    --dataset_name outfox \
    --prompts "/leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/en/outfox_GPT4.jsonl" \
    --output_path ./synthetic_llm_data/outfox_test_chat.jsonl \
    --human_key human_text \
    --huggingface_or_vllm vllm \
    --tensor_parallel_size 1 \
    --is_test true
