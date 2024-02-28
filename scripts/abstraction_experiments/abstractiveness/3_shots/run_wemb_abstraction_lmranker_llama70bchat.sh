#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --qos=normal
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=05:00:00
#SBATCH --output slurm_logs/data_modify-%j.out
#SBATCH --wait-all-nodes=1

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

srun python -u -m synthetic_llm_data.src.abstraction_pilot.lmranker_experiment \
    --data_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/pilot_dataset_en_utf8.csv" \
    --output_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/abstraction/3_shots/ranker_experiment/few_shots_ranker_test_70b.csv" \
    --temperature 0.8 \
    --max_batch_size 16 \
    --name_or_path "../models/hf_llama/llama-2-70b-chat-hf" \
    --preprocessing "abstraction_regression" \
    --project "wemb" \
    --huggingface_or_vllm "vllm" \
    --human_key "text" \
    --padding_side "left" \
    --min_new_tokens 10 \
    --max_new_tokens 20 \
    --tensor_parallel_size 4 \
    --max_seq_len 1 \
    --n_few_shots 3 \
    --selected_boundary 10000
