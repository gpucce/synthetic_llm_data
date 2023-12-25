import os
import json
from pathlib import Path

import torch
import datasets

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    # TopKLogitsWarper,
)

from ...utils import save_distributed_and_collect_on_main_rank

from .utils import (
    tokenize_and_mask,
    replace_masks,
    extract_fills,
    apply_extracted_fills,
    process_spaces,
    custom_parse_args,
    compute_loss
)


def tokenize_mask_extract_and_apply(
    ds, model, tok, col_name, args):

    n_modifications = args.n_modifications
    batch_size = args.batch_size

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
    output_path = Path(args.output_path)
    n_modifications = args.n_modifications
    modifier_model_name = args.modifier_model
    debug = args.debug
    model_name = (
        args.model_name if not debug 
        else "/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/tiny-random-llama")
    

    # ds = datasets.load_dataset("csv", data_files=data_path, delimiter="\t")
    ds = datasets.load_from_disk(data_path)

    ntasks   = int(os.environ.get("SLURM_NTASKS", 1))
    taskid  = int(os.environ.get("SLURM_PROCID", 0))
    localid = int(os.environ.get("SLURM_LOCALID", 0))

    if args.n_samples > 0:
        # df = df.iloc[: args.n_samples, :]
        # ds = ds.select(range(args.n_samples))
        for split in ds:
            ds[split] = ds[split].select(range(args.n_samples))

    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path / "experiment_modification_params.json", "w") as jf:
        json.dump(vars(args), jf)

    modifier_model = AutoModelForSeq2SeqLM.from_pretrained(
        modifier_model_name, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto")

    modifier_tok = AutoTokenizer.from_pretrained(
        modifier_model_name, fast=False, model_max_length=512 if not debug else 10)

    for split in ds:
        ds[split] = ds[split].shard(num_shards=int(ntasks), index=int(taskid))
        _ds = ds[split]

        for col_name in col_names.split(":"):

            _ds = _ds.map(
                lambda x: {
                    f"original_{col_name}": x[col_name], 
                    col_name: process_spaces(x[col_name]).replace("\n", " ")})

            _ds = _ds.map(
                lambda x: {col_name: x[col_name][:args.max_seq_len]
                           if not debug else x[col_name][:10]})

            _ds = tokenize_mask_extract_and_apply(
                ds=_ds, model=modifier_model, tok=modifier_tok,
                args=args, col_name=col_name)

            if not debug:
                count = 0
                while count < 10:
                    print(f"TRY: {count}")
                    mask = _ds.map(
                        lambda x: {"mask":
                            any([x[f"{col_name}_synthetic_{i}"] == ""
                                 for i in range(n_modifications)])})["mask"]

                    if not any(mask):
                        break

                    replacement = tokenize_mask_extract_and_apply(
                        _ds.filter(lambda _, idx: mask[idx], with_indices=True),
                        model=modifier_model, tok=modifier_tok, args=args, col_name=col_name)

                    idxs = [idx for idx, x in enumerate(mask) if x]
                    for i in range(replacement.num_rows):
                        for j in range(n_modifications):
                            _ds["modification_" + str(j)][idxs[i]] = \
                                replacement[i]["modification_" + str(j)]

                count += 1
        ds[split] = _ds
        del _ds

    del modifier_model
    del modifier_tok
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto")

    tok = AutoTokenizer.from_pretrained(
        model_name, fast=False, model_max_length=512 if not debug else 10)
    tok.pad_token_id = tok.eos_token_id

    for split in ds:
        _ds = ds[split]
        for col_name in col_names.split(":"):
            _ds = _ds.map(
                lambda x: {f"{col_name}_synthetic_{i}_loss":
                    compute_loss(x[f"{col_name}_synthetic_{i}"], model, tok)
                    for i in range(n_modifications)},
                batched=True, batch_size=args.batch_size)

        # ds.save_to_disk(output_path / "modifications_dataset")
        ds[split] = save_distributed_and_collect_on_main_rank(
            data_shard=_ds, global_rank=taskid, global_n_devices=ntasks,
            output_file=output_path, save_after_collect=False)

    if taskid == 0:
        ds.save_to_disk(output_path)


if __name__ == "__main__":
    main()
