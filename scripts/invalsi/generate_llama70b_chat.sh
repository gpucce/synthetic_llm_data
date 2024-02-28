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

export CUDA_VISIBLE_DEVICES="0,1,2,3"

# source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

# srun 
python -m synthetic_llm_data.src.invalsi.generation_experiment \
    --name_or_path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-70b-chat-hf/" \
    --output_path /leonardo_scratch/large/userexternal/gpuccett/datasets/ita_llm_data/invalsi_data/invalsi_mate_clean_predicted_llama70b_chat.csv \
    --data_path /leonardo_scratch/large/userexternal/gpuccett/datasets/ita_llm_data/invalsi_data/invalsi_mate_clean.csv \
    --seed 1 \
    --selected_boundary 1000 \
    --max_new_tokens 300 \
    --min_new_tokens 50 \
    --max_batch_size 16 \
    --max_seq_len 0 \
    --huggingface_or_vllm vllm \
    --use_beam_search True \
    --project invalsi \
    --preprocessing invalsi_mate \
    --human_key domanda \
    --padding_side left \
    --dtype float16

