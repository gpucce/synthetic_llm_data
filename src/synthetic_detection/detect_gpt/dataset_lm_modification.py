
import json
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
import datasets

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    # TopKLogitsWarper,
)

from .utils import (
    tokenize_and_mask,
    replace_masks,
    extract_fills,
    apply_extracted_fills,
    process_spaces,
    custom_parse_args,
)


def tokenize_mask_extract_and_apply(
    ds, model, tok, col_name, args):

    n_modifications = args.n_modifications
    batch_size = args.batch_size
    batched = batch_size > 1

    ds = ds.map(
        lambda x: {col_name: process_spaces(x[col_name]).replace("\n", " ")})
        
    ds = ds.map(
        lambda x: {f"modification_{i}":
            tokenize_and_mask(x[col_name], span_length=2, pct=args.pct_mask)
            for i in range(n_modifications)},
        batched=False, desc = "Tokenizing and masking")

    ds = ds.map(
        lambda x: {f"fills_{i}":
            replace_masks(x[f"modification_{i}"], model, tok, debug=args.debug)
            for i in range(n_modifications)},
        batched=True, batch_size=batch_size, desc = "Replacing masks")

    ds = ds.map(
        lambda x: {f"fills_{i}":
            extract_fills(x[f"fills_{i}"])
            for i in range(n_modifications)},
        batched=True, batch_size=batch_size, desc = "Extracting fills")

    ds = ds.map(
        lambda x: {f"{col_name}_synthetic_{i}":
            apply_extracted_fills(x[f"modification_{i}"], x[f"fills_{i}"])
            for i in range(n_modifications)},
        batched=True, batch_size=batch_size, desc = "Applying extracted fills")

    return ds

def main():
    args = custom_parse_args()

    data_path = args.data_path
    col_names = args.col_names
    output_path = args.output_path
    n_modifications = args.n_modifications
    model_name = args.modifier_model
    debug = args.debug

    # ds = datasets.load_dataset("csv", data_files=data_path, delimiter="\t")
    ds = datasets.load_from_disk(data_path)

    if args.n_samples > 0:
        # df = df.iloc[: args.n_samples, :]
        # ds = ds.select(range(args.n_samples))
        for split in ds:
            ds[split] = ds[split].select(range(args.n_samples))


    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    hftok = AutoTokenizer.from_pretrained(model_name, fast=False, model_max_length=512 if not debug else 10)

    for col_name in col_names.split(":"):
        # ds["train"][col_name] = ds["train"][col_name].apply(process_spaces).str.replace("\n", " ")

        start = time.time()

        for split in ds:
            ds[split] = tokenize_mask_extract_and_apply(
                ds=ds[split], model=model, tok=hftok, 
                args=args, col_name=col_name)

            count = 0
            while count < 10:
                mask = ds[split].map(
                    lambda x: {"mask":
                        any([x[f"modification_{i}"] == ""
                             for i in range(n_modifications)])})["mask"]

                if not any(mask):
                    break

                replacement = tokenize_mask_extract_and_apply(
                    ds[split].filter(mask), model=model, tok=hftok,
                    args=args, col_name=col_name)

                idxs = [idx for idx, x in enumerate(mask) if x]
                for i in range(replacement.num_rows):
                    for j in range(n_modifications):
                        ds[split][col_name][idxs[i]]["modification_" + str(j)] = \
                            replacement[i]["modification_" + str(j)]

                count += 1

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path.parent / "experiment_modification_params.json", "w") as jf:
        json.dump(vars(args), jf)
    ds.save_to_disk(output_path / "modifications_dataset")


if __name__ == "__main__":
    main()
