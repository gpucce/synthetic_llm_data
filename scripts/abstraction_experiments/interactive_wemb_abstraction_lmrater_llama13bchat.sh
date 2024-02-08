#!/bin/bash

python -u -m synthetic_llm_data.src.abstraction_pilot.lmrater_experiment \
    --data_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/pilot_dataset_en_utf8.csv" \
    --output_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/3_shots/few_shots_test_13b.csv" \
    --temperature 0.8 \
    --max_batch_size 16 \
    --name_or_path "../models/hf_llama/llama-2-13b-chat-hf" \
    --preprocessing "abstraction" \
    --project "wemb" \
    --human_key "text" \
    --padding_side "left" \
    --min_new_tokens 10 \
    --max_new_tokens 20 \
    --max_seq_len 1 \
    --n_few_shots 3


    # --data_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/pairs_abstraction_replacement_experiment_results_postprocessed.csv" \