#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --output slurm_logs/supervise_detect_hf_llama/llama-7b_change_it/hf_roberta/xlm-roberta-large/slurm.out

# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# cd /leonardo_scratch/large/userexternal/gpuccett/data
# source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

# srun 
python -m synthetic_llm_data.src.synthetic_detection.sentence_detection \
    --data_path /leonardo_scratch/large/userexternal/gpuccett/data/ita_synthetic_data/autoexperiment/change_it//hf_llama/llama-7b_change_it/ \
    --output_path /leonardo_scratch/large/userexternal/gpuccett/data/ita_supervised_synthetic_text_detection//hf_roberta/xlm-roberta-large_hf_llama/llama-7b_change_it \
    --model_name_or_path /leonardo_scratch/large/userexternal/gpuccett/models/hf_roberta/xlm-roberta-large \
    --col_name full_text \
    --max_samples 2000 \
    
