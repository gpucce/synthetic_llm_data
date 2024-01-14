import os
import copy
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
    compute_loglikelihood,
    batched_detect_score,
    custom_load_dataset,
)

from ...data_generation import generation

os.environ['OPENBLAS_NUM_THREADS'] = '64'
datasets.disable_caching()

def tokenize_mask_extract_and_apply(
    ds, model, tok, col_name, args, idx):

    batch_size = args.max_batch_size

    ds = ds.map(
        lambda x: {col_name: process_spaces(x[col_name]).replace("\n", " ")})

    ds = ds.map(
        lambda x: {f"modification_{idx}":
            tokenize_and_mask(x[col_name], span_length=2, pct=args.pct_mask)},
        batched=False, desc = "Tokenizing and masking")

    ds = ds.map(
        lambda x: {f"fills_{idx}":
            replace_masks(x[f"modification_{idx}"], model, tok)},
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

    ds = ds.remove_columns([f"fills_{idx}", f"modification_{idx}"])

    return ds, col_name

def main():
    args = custom_parse_args()


    col_names = args.col_names
    output_path = Path(args.output_path)
    n_modifications = args.n_modifications
    modifier_model_name = args.modifier_model
    debug = args.debug
    model_name = args.model_name

    ds = custom_load_dataset(args)
    ds = ds.filter(lambda x: len(x[col_names].split(" ")) > args.min_new_tokens + 30)

    if args.do_generation:
        assert args.dataset_type != "m4"

        ds = generation(args, ds, clean_temp_files=False, output_on_all_nodes=True)
        ds = datasets.DatasetDict({
            "train": datasets.Dataset.from_dict({
                col_names: ds["mixed_text"] + ds["human_text"],
                "label": [1] * len(ds["mixed_text"]) + [0] * len(ds["human_text"]),
                "prompt": ds["prompt"] * 2,
                f"human_{col_names}": ds["human_text"] * 2,
                "truncated_human_{col_names}": ds["truncated_human_text"] * 2,
                "cut_at_sentence": ds["cut_at_sentence"] * 2,
            })
        })

    ntasks   = int(os.environ.get("SLURM_NTASKS", 1))
    print(f"################### NTASKS {ntasks}")
    taskid  = int(os.environ.get("SLURM_PROCID", 0))
    print(f"################### TASKID {taskid}")
    localid = int(os.environ.get("SLURM_LOCALID", 0)) # Pylint: disable=unused-variable
    print(f"################### LOCALID {taskid}")



    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path / "experiment_modification_params.json", "w") as jf:
        json.dump(vars(args), jf)

    modifier_model = AutoModelForSeq2SeqLM.from_pretrained(
        modifier_model_name, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto")

    modifier_tok = AutoTokenizer.from_pretrained(
        modifier_model_name, fast=False, model_max_length=512)

    for split in ds:
        _ds = ds[split].shard(num_shards=ntasks, index=taskid)

        for col_name in col_names.split(":"):

            _ds = _ds.map(
                lambda x: {
                    f"original_{col_name}": x[col_name], 
                    col_name: process_spaces(x[col_name]).replace("\n", " ")})

            _ds = _ds.map(
                lambda x: {col_name: " ".join(
                    x[col_name].split(" ")[:args.max_seq_len])})

            for n_modif in range(n_modifications):
                _ds, local_col_name = tokenize_mask_extract_and_apply(
                    ds=_ds, model=modifier_model, tok=modifier_tok,
                    args=args, col_name=col_name, idx=n_modif)


                count = 0
                while count < 10:
                    count += 1
                    msg = f"SPLIT {split} COL_NAME {col_name} "
                    msg += f"MODIFICATION {n_modif} TRY: {count}"
                    print(msg)

                    if debug and count == 1:
                        import random
                        test_idxs = []
                        for _ in range(2):
                            debug_count = 0
                            test_idx = random.randint(0, _ds.num_rows)
                            while _ds[local_col_name][test_idx] == "" and debug_count < 10:
                                test_idx = random.randint(0, _ds.num_rows)
                                debug_count += 1
                            _ds = _ds.map(lambda x, idx:
                                {local_col_name: x[local_col_name] if idx != test_idx else ""},
                                with_indices=True, desc="Debug")
                            test_idxs.append(test_idx)

                    mask = _ds.map(lambda x: {"mask": x[local_col_name] == ""})["mask"]

                    if not any(mask):
                        break

                    idxs = [idx for idx, x in enumerate(mask) if x]

                    replacement, _ = tokenize_mask_extract_and_apply(
                        copy.deepcopy(_ds.select(idxs)), model=modifier_model,
                        tok=modifier_tok, args=args, col_name=col_name, idx=n_modif)
                    replacement = replacement[local_col_name]

                    if debug and count == 1:
                        _idxs = []
                        for test_idx in test_idxs:
                            assert test_idx in idxs
                            _idx = idxs.index(test_idx)
                            assert replacement[_idx] != _ds[local_col_name][test_idx]
                            print(f"############## TEST IDX {test_idx}")
                            print(replacement[_idx])
                            print(_ds[local_col_name][test_idx])
                            _idxs.append(_idx)

                    _ds = _ds.map(lambda x, idx:{local_col_name:
                        replacement[idxs.index(idx)] if idx in idxs
                        else x[local_col_name]}, with_indices=True)

                    if debug and count == 1:
                        for _idx, test_idx in zip(_idxs, test_idxs):
                            assert replacement[_idx] == _ds[local_col_name][test_idx]

        ds[split] = _ds
        del _ds

    del modifier_model
    del modifier_tok

    print(f"##################### NTASKS {ntasks} TASKID {taskid} MODIFICATION DONE")

    model = AutoModelForCausalLM.from_pretrained(
        model_name, # torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, device_map="auto")

    tok = AutoTokenizer.from_pretrained(
        model_name, fast=False, model_max_length=512)
    tok.pad_token_id = tok.eos_token_id

    for split in ds:
        _ds = ds[split]
        for col_name in col_names.split(":"):
            _ds = _ds.map(
                lambda x: {f"{col_name}_loss":
                            compute_loglikelihood(x[col_name], model, tok)},
                batched=True, batch_size=args.max_batch_size)

            for i in range(n_modifications):

                _ds = _ds.map(
                    lambda x: {f"{col_name}_synthetic_{i}_loss":
                               # remember the minus here
                                compute_loglikelihood(x[f"{col_name}_synthetic_{i}"], model, tok)},
                    batched=True, batch_size=args.max_batch_size)

                if i == 0 and debug:
                    print("DOC EXAMPLES", _ds[f"{col_name}"][:3])
                    print("ORIGINAL LOSS", _ds[f"{col_name}_loss"][:3])
                    print("MODIFIED DOC EXAMPLES", _ds[f"{col_name}_synthetic_0"][:3])
                    print("MODIFIED LOSS", _ds[f"{col_name}_synthetic_0_loss"][:3])

            _ds = _ds.map(
                lambda x: {f"{col_name}_score":
                            batched_detect_score(x, re_pattern="synthetic.*loss")},
                batched=True, batch_size=args.max_batch_size)

        ds[split] = save_distributed_and_collect_on_main_rank(
            data_shard=_ds, global_rank=taskid, global_n_devices=ntasks,
            split=split, output_file=output_path, save_after_collect=False)

    if taskid == 0:
        ds.save_to_disk(output_path)


if __name__ == "__main__":
    main()
