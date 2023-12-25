#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --qos=normal
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --output slurm_logs/data_modify-%j.out
#SBATCH --wait-all-nodes=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

cd /leonardo_scratch/large/userexternal/gpuccett/data/synthetic_llm_data

srun python -m src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --output-path "test_data_modification_output" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/t5-3b" \
    --data-path "/leonardo_scratch/large/userexternal/gpuccett/data/xsum" \
    --col-names "document" \
    --n-samples 128 \
    --batch-size 128 \
    --n-modifications 2