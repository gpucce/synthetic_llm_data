#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:30:00
#SBATCH --output slurm_logs/data_modify-%j.out
#SBATCH --wait-all-nodes=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -u -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --output-path "detectGPT_experiments/xsum/gpt2xlgen_t5-3bmodif_gpt2xldetect" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/t5-3b" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_gpt/gpt2-xl-hf" \
    --data-path "/leonardo_scratch/large/userexternal/gpuccett/datasets/xsum" \
    --name-or-path "/leonardo_scratch/large/userexternal/gpuccett/models/hf_gpt/gpt2-xl-hf" \
    --huggingface-or-vllm "huggingface" \
    --col-name "document" \
    --human-key "document" \
    --max-batch-size 16 \
    --do-generation \
    --do-modification \
    --dtype "float32" \
    --temperature 1.0 \
    --n-modifications 20 \
    --dataset-type disk \
    --min-new-tokens 150 \
    --max-new-tokens 200 \
    --max-seq-len 150 \
    --preprocessing xsum \
    --padding-side "left" \
    --n-samples 500 \
    --seed 15 \
    --do-compute-loss
