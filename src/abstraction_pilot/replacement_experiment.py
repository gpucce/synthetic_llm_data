import json
from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

from synthetic_llm_data.src.synthetic_detection.detect_gpt.utils import (
    process_spaces,
    tokenize_and_mask_with_indices,
    replace_masks,
    # replace_masks_with_many,
    extract_fills,
    apply_extracted_fills,
)

from .utils import postprocess_data, blow_columns
from .args import replacement_parse_args


def tokenize_mask_extract_and_apply(
    ds, model, tok, col_name, args, idx):

    batch_size = args.max_batch_size

    ds = ds.map(
        lambda x: {f"modification_{idx}":
            tokenize_and_mask_with_indices(x[col_name], x["begin"], x["end"])},
        batched=False, desc = "Tokenizing and masking")

    ds = ds.map(
        lambda x: {f"fills_{idx}":
            replace_masks(x[f"modification_{idx}"], model, tok, args)},
            # replace_masks_with_many(x[f"modification_{idx}"], model, tok, temperature=0.8)},
        batched=True, batch_size=batch_size, desc = "Replacing masks")

    ds = ds.map(
        lambda x: {f"fills_{idx}":
            extract_fills(x[f"fills_{idx}"])},
        batched=True, batch_size=batch_size, desc = "Extracting fills")

    col_name = f"{col_name}_synthetic_{idx}"

    ds = ds.map(
        lambda x: {col_name:
            apply_extracted_fills(x[f"modification_{idx}"], x[f"fills_{idx}"])},
        batched=True, batch_size=batch_size, desc = "Applying extracted fills")

    ds = ds.map(
        lambda x: {col_name: process_spaces(x[col_name]).replace("\n", " ")})

    ds = ds.remove_columns([f"fills_{idx}", f"modification_{idx}"])

    return ds

def main(args):

    print("Loading data...")
    model_name_or_path = args.name_or_path
    col_name = args.col_name
    n_modifications = args.n_modifications

    ds = load_dataset("csv", data_files=args.data_path)["train"]

    if args.max_prompts is not None:
        ds = ds.shuffle(args.seed).select(range(args.max_prompts))

    print("Loading model...")
    modifier_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path, low_cpu_mem_usage=True, device_map="auto")

    modifier_tok = AutoTokenizer.from_pretrained(
        model_name_or_path, fast=False, model_max_length=512)

    print("Modifying data...")
    for idx in range(n_modifications):
        ds = tokenize_mask_extract_and_apply(
            ds, modifier_model, modifier_tok, col_name, args, idx)

    ds = postprocess_data(ds)
    blown_df = blow_columns(ds.to_pandas())

    ds.to_csv(args.output_path)
    blown_df.to_csv(
        Path(args.output_path).parent / ("blown_" + Path(args.output_path).name), index=False)

    with open(Path(args.output_path).parent / "experiment_params.json", "w") as f:
        f.write(json.dumps(vars(args)))

if __name__ == "__main__":
    main(replacement_parse_args())
