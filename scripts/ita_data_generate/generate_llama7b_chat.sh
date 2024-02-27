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

# source /leonardo_scratch/large/userexternal/gpuccett/datasets/data_venv/bin/activate

# srun 
python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --data-path /leonardo_scratch/large/userexternal/gpuccett/datasets/CHANGE-it/test.jsonl \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-chat-hf/" \
    --output-path /leonardo_scratch/large/userexternal/gpuccett/datasets/ita_synthetic_data/llama7b_chat_change_it_vllm \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-chat-hf/" \
    --seed 1 \
    --col-name full_text \
    --max-new-tokens 400 \
    --min-new-tokens 400 \
    --max-batch-size 16 \
    --max-seq-len 200 \
    --n-samples 100 \
    --huggingface-or-vllm vllm \
    --use-beam-search True \
    --project ita_news \
    --preprocessing change_it \
    --human-key headline \
    --human-text-key full_text \
    --do-generation \
    --dataset-type json \
    --length-filter 0 \
    --padding-side left \
    --dtype float16

