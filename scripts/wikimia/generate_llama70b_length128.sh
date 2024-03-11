#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output slurm_logs/generate-%j.out

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-70b-hf/" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-70b-hf/" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/t5-3b" \
    --data-path /leonardo_scratch/large/userexternal/gpuccett/datasets/wikimia/WikiMIA_length128.json \
    --output-path /leonardo_scratch/large/userexternal/gpuccett/data/wikimia_experiments/llama-2-70b_length128 \
    --seed 1 \
    --col-name input \
    --max-new-tokens 250 \
    --min-new-tokens 200 \
    --max-batch-size 8 \
    --max-seq-len 150 \
    --n-samples 250 \
    --selected-boundary 30 \
    --huggingface-or-vllm huggingface \
    --project pre_gen \
    --preprocessing wikimia \
    --human-key input \
    --do-generation \
    --dataset-type json \
    --length-filter 0 \
    --padding-side left \
    --dtype float16 \
    --do-modification \
    --n-modifications 20 \
    --temperature 1.0 \
    --do-compute-loss


