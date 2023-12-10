#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --output slurm_logs/data_complete-%j.out
#SBATCH --wait-all-nodes=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -m synthetic_llm_data.src.data_complete \
    --output_file "test_data_complete_output" \
    --model_name_or_path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-chat-hf" \
    --is_test False
