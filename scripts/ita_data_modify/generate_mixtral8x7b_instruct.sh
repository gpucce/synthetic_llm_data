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

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/datasets/data_venv/bin/activate

# srun 

python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_mistral/Mixtral-8x7B-Instruct-v0.1/" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_mistral/Mixtral-8x7B-Instruct-v0.1/" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/it5" \
    --data-path /leonardo_scratch/large/userexternal/gpuccett/datasets/CHANGE-it/test.jsonl \
    --output-path /leonardo_scratch/large/userexternal/gpuccett/datasets/ita_synthetic_data/mixtral8x7b_instruct_v01_change_it_vllm \
    --seed 1 \
    --human-key full_text \
    --col-name full_text \
    --length-filter 100 \
    --max-new-tokens 250 \
    --min-new-tokens 200 \
    --max-batch-size 16 \
    --max-seq-len 0 \
    --n-samples 100 \
    --huggingface-or-vllm huggingface \
    --use-beam-search True \
    --project ita_news \
    --preprocessing change_it \
    --do-generation \
    --dataset-type json \
    --length-filter 0 \
    --padding-side left \
    --dtype float16 \
    --do-modification \
    --n-modifications 2 \
    --temperature 1.0 \
    --do-compute-loss

