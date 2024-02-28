
module load openmpi

export CUDA_VISIBLE_DEVICES="0,1,2,3"

source /leonardo_scratch/large/userexternal/gpuccett/data/data_venv/bin/activate

mpirun python -u -m synthetic_llm_data.src.synthetic_detection.detect_gpt.dataset_lm_modification \
    --output-path "test_data_modification_output_mpirun" \
    --modifier-model "/leonardo_scratch/large/userexternal/gpuccett/models/hf_t5/t5-3b" \
    --model-name "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-chat-hf" \
    --data-path "/leonardo_scratch/large/userexternal/gpuccett/datasets/xsum" \
    --col-names "document" \
    --n-samples 128 \
    --batch-size 32 \
    --n-modifications 2 \
    --debug