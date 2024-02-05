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
python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-chat-hf/" \
    --output-path /leonardo_scratch/large/userexternal/gpuccett/data/ita_llm_data/invalsi_data/invalsi_mate_clean_predicted_llama7b_chat \
    --data-path /leonardo_scratch/large/userexternal/gpuccett/data/ita_llm_data/invalsi_data/invalsi_mate_clean.csv \
    --dataset-type csv \
    --seed 1 \
    --selected-boundary 1000 \
    --length-filter 0 \
    --col-name domanda \
    --max-new-tokens 300 \
    --min-new-tokens 50 \
    --max-batch-size 16 \
    --max-seq-len 0 \
    --huggingface-or-vllm huggingface \
    --use-beam-search True \
    --project invalsi \
    --preprocessing invalsi_mate \
    --human-key domanda \
    --do-generation \
    --length-filter 0 \
    --padding-side left \
    --dtype float16

