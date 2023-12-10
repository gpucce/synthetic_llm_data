#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
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
    --model_name_or_path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-70b-chat-hf" \
    --is_test False \
    --max_samples 100
