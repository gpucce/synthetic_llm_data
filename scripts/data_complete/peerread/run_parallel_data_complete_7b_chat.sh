#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --qos=normal
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --output slurm_logs/data_complete-%j.out
#SBATCH --wait-all-nodes=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -m synthetic_llm_data.src.data_generation.data_complete \
    --output_path "test_data_complete_output" \
    --name_or_path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-chat-hf" \
    --is_test False \
    --huggingface_or_vllm "vllm" \
    --max_batch_size 32 \
    --base_path "/leonardo_scratch/large/userexternal/gpuccett/data/semeval2024-private/semeval-taskC/data/" \
    --split_names "train" "dev" "test" \
    --split_files "train/train_chatgpt.csv" "dev/dev_chatgpt.csv" "test/test_chatgpt.csv" \
    --human_key "full_human_review"


