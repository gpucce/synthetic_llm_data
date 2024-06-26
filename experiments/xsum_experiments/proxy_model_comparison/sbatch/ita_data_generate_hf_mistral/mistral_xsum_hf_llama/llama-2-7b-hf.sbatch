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
#SBATCH --output slurm_logs/set7b/ita_data_generate_hf_mistral/mistral_xsum_hf_llama/llama-2-7b-hf/slurm.out

export CUDA_VISIBLE_DEVICES="0,1,2,3"

cd /leonardo_scratch/large/userexternal/gpuccett/data
source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --model-name /leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-hf \
    --data-path /leonardo_scratch/large/userexternal/gpuccett/data/ita_synthetic_data/autoexperiment/xsum/hf_mistral/mistral_xsum \
    --output-path /leonardo_scratch/large/userexternal/gpuccett/data/ita_synthetic_data/autoexperiment/xsum/comparisons/hf_mistral/mistral_xsum/hf_llama/llama-2-7b-hf \
    --seed 1 \
    --human-key document \
    --col-name document \
    --max-batch-size 16 \
    --huggingface-or-vllm huggingface \
    --dataset-type disk \
    --length-filter 0 \
    --dtype float16 \
    --n-modifications 20 \
    --temperature 1.0 \
    --do-compute-loss
