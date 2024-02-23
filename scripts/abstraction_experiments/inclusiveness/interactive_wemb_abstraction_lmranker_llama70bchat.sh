#!/bin/bash

python -u -m synthetic_llm_data.src.abstraction_pilot.lmranker_experiment \
    --data_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/pilot_dataset_en_utf8.csv" \
    --output_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/inclusiveness/3_shots/few_shots_ranker_test_70b.csv" \
    --temperature 0.8 \
    --max_batch_size 16 \
    --name_or_path "../models/hf_llama/llama-2-70b-chat-hf" \
    --preprocessing "inclusiveness_regression" \
    --project "wemb" \
    --huggingface_or_vllm "huggingface" \
    --human_key "text" \
    --padding_side "left" \
    --min_new_tokens 10 \
    --max_new_tokens 20 \
    --tensor_parallel_size 4 \
    --max_seq_len 1 \
    --n_few_shots 3 \
    --selected_boundary 10000
