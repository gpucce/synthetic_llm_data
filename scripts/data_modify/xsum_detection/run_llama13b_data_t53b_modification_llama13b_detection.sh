#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-task=2
#SBATCH --cpus-per-task=16
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --qos=normal
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=12:00:00
#SBATCH --output slurm_logs/data_modify-%j.out
#SBATCH --wait-all-nodes=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -u -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-13b-hf" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/t5-3b" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-13b-hf" \
    --data-path "/leonardo_scratch/large/userexternal/gpuccett/data/xsum" \
    --output-path "detectGPT_experiments/xsum/llama13bgen_t5-3bmodif_llama13bdetect_huggingface" \
    --col-name "document" \
    --human-key "document" \
    --max-batch-size 16 \
    --do-generation \
    --do-modification \
    --n-modifications 50 \
    --dataset-type disk \
    --min-new-tokens 150 \
    --max-new-tokens 200 \
    --huggingface-or-vllm huggingface \
    --max-seq-len 150 \
    --preprocessing xsum \
    --padding-side "left" \
    --n-samples 1000 \
    --temperature 1.0 \
    --seed 10 \
    --do-compute-loss
