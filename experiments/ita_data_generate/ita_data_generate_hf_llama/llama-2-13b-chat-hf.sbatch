#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=2
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output slurm_logs/set13b/ita_data_generate_hf_llama/llama-2-13b-chat-hf/slurm.out

export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd /leonardo_scratch/large/userexternal/gpuccett/data
source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --name-or-path /leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-13b-chat-hf \
    --model-name /leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-13b-chat-hf \
    --modifier-model /leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/it5 \
    --data-path /leonardo_scratch/large/userexternal/gpuccett/data/CHANGE-it/test.jsonl \
    --output-path /leonardo_scratch/large/userexternal/gpuccett/data/ita_synthetic_data/autoexperiment/change_it/hf_llama/llama-2-13b-chat-hf \
    --seed 1 \
    --human-key full_text \
    --col-name full_text \
    --max-new-tokens 250 \
    --min-new-tokens 200 \
    --max-batch-size 16 \
    --max-seq-len 150 \
    --n-samples 5000 \
    --huggingface-or-vllm huggingface \
    --project ita_news \
    --preprocessing change_it \
    --do-generation \
    --dataset-type json \
    --length-filter 100 \
    --padding-side left \
    --dtype float16 \
    --do-modification \
    --n-modifications 20 \
    --temperature 1.0 \
    --do-compute-loss \
    --tensor-parallel-size 2


