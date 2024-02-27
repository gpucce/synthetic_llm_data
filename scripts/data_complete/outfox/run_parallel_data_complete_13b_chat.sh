#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --output slurm_logs/generate-%j.out

module purge 
module load gcc
module load cuda

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/datasets/data_venv/bin/activate

srun python -m synthetic_llm_data.src.data_generation.data_complete \
    --name_or_path /leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-13b-chat-hf/ \
    --seed 1 \
    --max_new_tokens 384 \
    --max_batch_size 16 \
    --use_beam_search True \
    --huggingface_or_vllm "vllm" \
    --output_path /leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/outfox_llama13b_chat.csv \
    --tensor_parallel_size 2 \
    --preprocessing outfox \
    --base_path "/leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/" \
    --split_names train \
    --split_files outfox_gpt4.csv \
    --human_key "human_text" \
    --columns_to_remove "machine_review" "prompts" "partial_essay"
