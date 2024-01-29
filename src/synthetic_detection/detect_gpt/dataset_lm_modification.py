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

from ...utils import save_distributed_and_collect_on_main_rank, save_to_right_format

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
    multiple_length_cutting
)

from ...data_generation import generation

datasets.disable_caching()


def tokenize_mask_extract_and_apply(
    ds, model, tok, col_name, args, idx):

    batch_size = args.max_batch_size

    ds = ds.map(
        lambda x: {f"modification_{idx}":
            tokenize_and_mask(x[col_name], span_length=2, pct=args.pct_mask)},
        batched=False, desc = "Tokenizing and masking")

    ds = ds.map(
        lambda x: {f"fills_{idx}":
            replace_masks(x[f"modification_{idx}"], model, tok, args)},
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

    cols_to_remove = [f"fills_{idx}", f"modification_{idx}"]
    cols_to_remove = [i for i in cols_to_remove if i in ds.column_names]
    ds = ds.remove_columns(cols_to_remove)

    return ds, col_name

def cut_to_shortest(x, args, col_names):
    out = {}
    min_len = min(len(x[col_name].split(" ")) for col_name in col_names)
    min_len = min(min_len, args.max_seq_len)
    for col_name in col_names:
        out[col_name] = " ".join(x[col_name].split(" ")[:min_len])
    return out

def main(args):

    assert not args.split_at_random_length, "Split at random length messes experiments"
    col_name = args.col_name
    output_path = Path(args.output_path)
    n_modifications = args.n_modifications
    debug = args.debug
    modifier_model_name = args.modifier_model
    model_name = args.model_name

    ntasks = int(os.environ.get("SLURM_NTASKS", 1))
    print(f"################### NTASKS {ntasks}")
    taskid = int(os.environ.get("SLURM_PROCID", 0))
    print(f"################### TASKID {taskid}")
    localid = int(os.environ.get("SLURM_LOCALID", 0)) # pylint: disable=unused-variable
    print(f"################### LOCALID {localid}")


    ds = custom_load_dataset(args)
    ds = ds.select(range(min(args.n_samples * 10, ds.num_rows)))
    # remove short docs
    ds = ds.filter(
        lambda x: len(x[col_name].split(" ")) > args.length_filter)
    ds = ds.select(range(args.n_samples))    

    ds = ds.shard(num_shards=ntasks, index=taskid)

    output_path.mkdir(exist_ok=True, parents=True)
    with open(output_path / "experiment_modification_params.json", "w") as jf:
        json.dump(vars(args), jf)

    if args.do_generation:
        assert args.dataset_type != "m4"

        ds = ds.shuffle(args.seed)
        ds = ds.map(lambda x: {col_name: process_spaces(x[col_name]).strip()})
        ds = ds.map(lambda x: {col_name: ' '.join(x[col_name].split())})

        ds = generation(args, ds)
        ds = ds.map(lambda x: cut_to_shortest(x, args, ["human_text", "mixed_text"]))
        ds.to_json(str(output_path / "paired_ds.json"))
        future_df = {
                col_name: ds["mixed_text"] + ds["human_text"],
                "label": [1] * len(ds["mixed_text"]) + [0] * len(ds["human_text"]),
                "prompt": ds["prompt"] * 2,
                f"human_{col_name}": ds["human_text"] * 2,
                f"truncated_human_{col_name}": ds["truncated_human_text"] * 2,
                f"truncated_machine_{col_name}": ds["machine_text"] * 2,
                "cut_at_sentence": ds["cut_at_sentence"] * 2,
            }

        if "headline" in ds:
            future_df["headline"] = ds["headline"] * 2

        ds = datasets.Dataset.from_dict(future_df)

    gen_done_msg = f"################### NTASKS {ntasks} TASKID {taskid} GENERATION DONE"
    if not args.do_generation:
        gen_done_msg = gen_done_msg.replace("DONE", "SKIPPED")
    print(gen_done_msg)

    if args.multi_length_clipping:
        ds = ds.map(lambda x: multiple_length_cutting(x, col_name, args.max_seq_len))

    if args.do_modification:
        modifier_model = AutoModelForSeq2SeqLM.from_pretrained(
            modifier_model_name, low_cpu_mem_usage=True, device_map="auto")

        modifier_tok = AutoTokenizer.from_pretrained(
            modifier_model_name, fast=False, model_max_length=1024)

        for n_modif in range(n_modifications):
            ds, local_col_name = tokenize_mask_extract_and_apply(
                ds=ds, model=modifier_model, tok=modifier_tok,
                args=args, col_name=col_name, idx=n_modif)

            count = 0
            while count < 10:
                count += 1
                msg = f"COL_NAME {col_name} "
                msg += f"MODIFICATION {n_modif} TRY: {count}"
                print(msg)

                if debug and count == 1:
                    import random
                    test_idxs = []
                    for _ in range(2):
                        debug_count = 0
                        test_idx = random.randint(0, ds.num_rows)
                        while ds[local_col_name][test_idx] == "" and debug_count < 10:
                            test_idx = random.randint(0, ds.num_rows)
                            debug_count += 1
                        ds = ds.map(lambda x, idx:
                            {local_col_name: x[local_col_name] if idx != test_idx else ""},
                            with_indices=True, desc="Debug")
                        test_idxs.append(test_idx)

                mask = ds.map(lambda x: {"mask": x[local_col_name] == ""})["mask"]
                if not any(mask):
                    break

                idxs = [idx for idx, x in enumerate(mask) if x]
                print(idxs)

                replacement, _ = tokenize_mask_extract_and_apply(
                    copy.deepcopy(ds.select(idxs)), model=modifier_model,
                    tok=modifier_tok, args=args, col_name=col_name, idx=n_modif)
                replacement = replacement[local_col_name]
                print(replacement)

                if debug and count == 1:
                    _idxs = []
                    for test_idx in test_idxs:
                        assert test_idx in idxs
                        _idx = idxs.index(test_idx)
                        assert replacement[_idx] != ds[local_col_name][test_idx]
                        print(f"############## TEST IDX {test_idx}")
                        print(replacement[_idx])
                        print(ds[local_col_name][test_idx])
                        _idxs.append(_idx)

                ds = ds.map(lambda x, idx:{local_col_name:
                    replacement[idxs.index(idx)] if idx in idxs
                    else x[local_col_name]}, with_indices=True)

                if debug and count == 1:
                    for _idx, test_idx in zip(_idxs, test_idxs):
                        assert replacement[_idx] == ds[local_col_name][test_idx]

        del modifier_model
        del modifier_tok

    mod_done_msg = f"################### NTASKS {ntasks} TASKID {taskid} MODIFICATION DONE"
    if not args.do_modification:
        mod_done_msg = mod_done_msg.replace("DONE", "SKIPPED")
    print(mod_done_msg)

    if args.do_compute_loss:
        dtype_map = {
            "bfloat16": torch.bfloat16, 
            "float16": torch.float16, 
            "float32": torch.float32}
        dtype = dtype_map[args.dtype]

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype,
            low_cpu_mem_usage=True, device_map="auto")

        tok = AutoTokenizer.from_pretrained(
            model_name, fast=False, model_max_length=1024)

        if not hasattr(tok, "pad_token_id") or tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id

        ds = ds.map(
            lambda x: {f"{col_name}_loss":
                        compute_loglikelihood(x[col_name], model, tok)},
            batched=True, batch_size=args.max_batch_size)

        for i in range(n_modifications):
            ds = ds.map(
                lambda x: {f"{col_name}_synthetic_{i}_loss":
                    compute_loglikelihood(x[f"{col_name}_synthetic_{i}"], model, tok)},
                batched=True, batch_size=args.max_batch_size, desc="Compute loglikelihood")

            print(f"LOSS {i} DONE")
            if i == 0:
                print("DOC EXAMPLES", ds[f"{col_name}"][:3])
                print("ORIGINAL LOSS", ds[f"{col_name}_loss"][:3])
                print("MODIFIED DOC EXAMPLES", ds[f"{col_name}_synthetic_0"][:3])
                print("MODIFIED LOSS", ds[f"{col_name}_synthetic_0_loss"][:3])

        ds = ds.map(
            lambda x: {f"{col_name}_score":
                batched_detect_score(x, col_name=f"{col_name}_loss", re_pattern="synthetic.*loss")},
            batched=True, batch_size=args.max_batch_size, desc="Compute detectGPT")

    loss_done_msg = f"################### NTASKS {ntasks} TASKID {taskid} LOSS COMPUTATION DONE"
    if not args.do_compute_loss:
        loss_done_msg = loss_done_msg.replace("DONE", "SKIPPED")
    print(loss_done_msg)

    ds = save_distributed_and_collect_on_main_rank(
        data_shard=ds, global_rank=taskid, global_n_devices=ntasks,
        output_file=output_path, save_after_collect=False)

    if taskid == 0:
        ds.save_to_disk(output_path)
        save_to_right_format(ds, output_path / "json_data_version.json")

    print("EVERYTHING DONE")

if __name__ == "__main__":
    main(custom_parse_args())
 