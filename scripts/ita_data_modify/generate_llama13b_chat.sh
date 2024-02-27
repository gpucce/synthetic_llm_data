#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output slurm_logs/generate-%j.out

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/datasets/data_venv/bin/activate

# srun 

python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-13b-chat-hf/" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-13b-chat-hf/" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/it5" \
    --data-path /leonardo_scratch/large/userexternal/gpuccett/datasets/CHANGE-it/test.jsonl \
    --output-path /leonardo_scratch/large/userexternal/gpuccett/datasets/ita_synthetic_data/llama_13b_chat_vllm \
    --seed 1 \
    --col-name full_text \
    --max-new-tokens 250 \
    --min-new-tokens 200 \
    --max-batch-size 8 \
    --max-seq-len 150 \
    --n-samples 100 \
    --selected-boundary 100 \
    --huggingface-or-vllm huggingface \
    --project ita_news \
    --preprocessing change_it \
    --human-key full_text \
    --do-generation \
    --dataset-type json \
    --length-filter 0 \
    --padding-side left \
    --dtype float16 \
    --do-modification \
    --n-modifications 2 \
    --temperature 1.0 \
    --do-compute-loss


