#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
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
    --name_or_path /leonardo_scratch/large/userexternal/gpuccett/models/hf_gpt/gpt2-hf \
    --seed 1 \
    --is_test "true" \
    --use_beam_search True \
    --huggingface_or_vllm "vllm" \
    --output_path ./data_complete_test \
    --tensor_parallel_size 1 \
    --preprocessing outfox \
    --base_path "/leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/" \
    --split_names train \
    --split_files outfox_gpt4.csv \
    --human_key "human_text"
