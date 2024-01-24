#!/bin/bash

python -u -m synthetic_llm_data.src.abstraction_pilot.replacement_experiment \
    --data_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/pilot_dataset_en_utf8.csv" \
    --output_path "/leonardo_scratch/large/userexternal/gpuccett/data/wemb_abstraction_data/abstraction_replacement_test.csv" \
    --n_modifications 2 \
    --temperature 0.8 \
    --max_batch_size 32 \
    --name_or_path "../models/hf_t5/t5-small" \
    --col_name "text"

