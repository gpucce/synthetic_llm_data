#!/bin/bash

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --output slurm_logs/set7b/data_generate_hf_llama/llama-7b_xsum/slurm.out

export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd /leonardo_scratch/large/userexternal/gpuccett/data
source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --name-or-path /leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-7b_xsum \
    --model-name /leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-7b_xsum \
    --modifier-model /leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/t5-3b \
    --data-path /leonardo_scratch/large/userexternal/gpuccett/data/xsum/test.jsonl \
    --output-path /leonardo_scratch/large/userexternal/gpuccett/data/ita_synthetic_data/autoexperiment/xsum/hf_llama/llama-7b_xsum \
    --seed 1 \
    --human-key document \
    --col-name document \
    --max-new-tokens 250 \
    --min-new-tokens 200 \
    --max-batch-size 16 \
    --max-seq-len 150 \
    --n-samples 1000 \
    --huggingface-or-vllm huggingface \
    --project semeval_task_3 \
    --preprocessing xsum \
    --do-generation \
    --dataset-type json \
    --length-filter 100 \
    --padding-side left \
    --dtype float16 \
    --do-modification \
    --n-modifications 20 \
    --temperature 1.0 \
    --do-compute-loss \
    --tensor-parallel-size 1


