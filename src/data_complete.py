"""Complete a given text"""
import pdb
import json
import re
import random
import os
from argparse import ArgumentParser
from tqdm.auto import tqdm
# from nltk import sent_tokenize
import transformers as ts
import datasets
import torch
from pathlib import Path

from .utils import (
    read_PeerRead, 
    str2bool,
    save_distributed_and_collect_on_main_rank
)

# old_prompt = """Task:
# Complete a partially-written peer review of the paper.

# Make sure that:
# 1. The completion is of at least {num_of_words} words.
# 2. You only output the completion of the partial review and not write it from the beginning.

# Title of the paper:
# {paper_title}

# Abstract of the paper:
# {paper_abstract}

# Partially-written review:
# \"\"\"
# {partial_review}
# \"\"\"

# Continuation:
# """

prompt = """Title of the paper:
{paper_title}

Abstract of the paper:
{paper_abstract}

Review:
{partial_review}
"""

def parse_completion(completion):
    return json.loads(completion)["completion"]

def split_by_full_stop(text):
    return re.split(r"(?<![A-Z\d])[.!?] +(?=[A-Z])", text)

def get_prompt(data_item, model_name):
    """
    A part of human review could be cut off 
    1. randomly at a word present in 1/10th to 5/10th of the review.
    2. randomly at a closest sentence boundary in the 1/10th to 5/10th of the review.
    """

    human_review = data_item["comment"]
    cut_at_sentence = random.randint(0, 2) > 1

    sentences = split_by_full_stop(human_review)
    words = human_review.split(" ")

    num_of_words = len(words)
    sentence_boundaries = [0] + [len(sentence.split(" ")) for sentence in sentences]
    sentence_boundaries = [
        sum(sentence_boundaries[:i]) for i in range(1, len(sentence_boundaries))
    ]

    # selecting human review from 1/10th to the 5/10th of the review
    selected_human_review_word = random.randint(
        int(num_of_words / 10), int(num_of_words / 2)
    )
    selected_human_review = words[:selected_human_review_word]

    if cut_at_sentence:
        # get the closest sentence boundary
        distance_to_sentence_boundary = [
            abs(len(selected_human_review) - boundary)
            for boundary in sentence_boundaries
        ]
        selected_boundary = sentence_boundaries[
            distance_to_sentence_boundary.index(min(distance_to_sentence_boundary))
        ]
    else:
        selected_boundary = selected_human_review_word

    num_of_words_to_generate = num_of_words - selected_boundary
    partial_review = words[:selected_boundary]
    partial_review = " ".join(partial_review)

    updated_prompt = prompt.format(
        paper_title=data_item["title"],
        paper_abstract=data_item["abstract"],
        partial_review=partial_review,
        num_of_words=num_of_words_to_generate,
    )

    data_item["machine_text"] = updated_prompt
    data_item["human_text"] = prompt.format(
        paper_title=data_item["title"],
        paper_abstract=data_item["abstract"],
        partial_review=human_review, 
        num_of_words=num_of_words_to_generate
    )
    
    data_item["human_end_boundaries"] = selected_boundary
    data_item["cut_at_sentence"] = cut_at_sentence

    keys_to_pop = ["chatgpt_reviews", "davinci_reviews", "prompts"]
    for key in keys_to_pop:
        data_item.pop(key, None)

    return data_item


def generate(text, model, tokenizer, is_test=False):
    if is_test:
        text = [i[:20] for i in text]
    tokenizer.pad_token = tokenizer.eos_token
    ids = tokenizer(
        text,
        truncation=True,
        return_tensors='pt',
        padding=True,
        max_length=2048,
    )
    out = model.generate(
        **{i:j.to(model.device) for i,j in ids.items()},
        # min_new_tokens=10,
        max_new_tokens=max(ids['input_ids'].shape[1], 512),
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
    return parser.parse_args()

def main(args):


    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))
    is_test = args.is_test
    
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
    device = torch.device(local_rank)
    print('Loading the model...')
    
    model_dir = Path('/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/')
    model_path = 'tiny-random-llama' if is_test else 'llama-2-7b-hf'
    checkpoint = str(model_dir / model_path)
    
    model = ts.AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.to(device)

    tokenizer = ts.AutoTokenizer.from_pretrained(checkpoint)
    print('Model successfully loaded.')

    data = data.map(lambda x: get_prompt(x, Path(checkpoint).name), desc="Generating prompts")

    data = data.map(
        lambda x: {
            i:j if i != "machine_text" 
            else generate(j, model, tokenizer, is_test=is_test) 
            for i,j in x.items()
        }, desc="Generating machine text", batched=True, batch_size=2 if is_test else 8)

    if "output_file" not in args:
        args.output_file = f"/leonardo_scratch/large/userexternal/gpuccett/data/PeerRead_synthetic_continuation"
    
    save_distributed_and_collect_on_main_rank(
        data_shard=data, args=args, global_rank=global_rank, global_n_devices=world_size
    )
    
    
    


if __name__ == "__main__":
    main(parse_args())