"""Complete a given text"""

import os
from argparse import ArgumentParser
from pathlib import Path
# from nltk import sent_tokenize
import transformers as ts
import datasets
import torch

from .preprocessing import get_semeval_task3_prompt
from .utils import (str2bool, save_distributed_and_collect_on_main_rank)

datasets.disable_caching()

def generate(text, model, tokenizer, is_test=False):
    _max_new_tokens = 2048
    if is_test:
        text = [i[:20] for i in text]
        _max_new_tokens = 512

    tokenizer.pad_token = tokenizer.eos_token
    ids = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
    )

    max_new_tokens = max(ids['input_ids'].shape[1], _max_new_tokens)

    out = model.generate(
        **{i:j.to(model.device) for i,j in ids.items()},
        max_new_tokens=max_new_tokens,
        # min_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        no_repeat_ngram_size=6,
        repetition_penalty=1.05,
    )
    return tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--is_test", type=str2bool, default=True)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()

def main(args):

    is_test = args.is_test

    # local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    # device = torch.device(local_rank)

    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    print("Loading the data...")

    # HF dataset
    # data = read_PeerRead("/leonardo_scratch/large/userexternal/gpuccett/data/PeerRead/")
    data_files = "/leonardo_scratch/large/userexternal/gpuccett/data/PeerRead_full.jsonl"
    data = datasets.load_dataset(
        "json", data_files=data_files)

    # TODO: change this
    if "train" in data:
        data = data["train"]

    if is_test and args.max_samples is None:
        args.max_samples = 100

    if args.max_samples is not None:
        data = data.select(range(args.max_samples))

    data = data.shard(num_shards=world_size, index=global_rank)

    print('Loading the model...')

    if is_test:
        if args.model_name_or_path is None:
            model_dir = Path('/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/')
            model_path = "tiny-random-llama" if is_test else "llama-2-7b-hf"
            args.model_name_or_path = str(model_dir / model_path)

    checkpoint = args.model_name_or_path

    model = ts.AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map="auto")

    tokenizer = ts.AutoTokenizer.from_pretrained(checkpoint)
    print('Model successfully loaded.')

    model_name = Path(checkpoint).name
    data = data.map(lambda x: get_semeval_task3_prompt(x, model_name), desc="Generating prompts")

    args.batch_size = 2 if is_test else args.batch_size
    data = data.map(
        lambda x: {
            i:j if i != "machine_text"
            else generate(j, model, tokenizer, is_test=is_test)
            for i,j in x.items()
        }, desc="Generating machine text", batched=True, batch_size=args.batch_size)

    if "output_file" not in args:
        args.output_file = "/leonardo_scratch/large/userexternal/" + \
        "gpuccett/data/PeerRead_synthetic_continuation"
    args.output_file += f"_model_{model_name}"

    save_distributed_and_collect_on_main_rank(
        data_shard=data, args=args, global_rank=global_rank, global_n_devices=world_size
    )


if __name__ == "__main__":
    main(parse_args())
