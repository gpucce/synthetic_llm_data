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
#SBATCH --time=02:00:00
#SBATCH --output slurm_logs/data_modify-%j.out
#SBATCH --wait-all-nodes=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -u -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --output-path "ita_detectGPT_camoscio/camoscio70gen_it5modif_llama7bdetect" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/it5" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-chat-hf" \
    --data-path "/leonardo_scratch/large/userexternal/gpuccett/datasets/semeval2024-private/data/other_languages/italian_news_camoscio.jsonl" \
    --col-names "document" \
    --batch-size 32 \
    --n-modifications 50 \
    --max-seq-len 256 \
    --n-samples 500