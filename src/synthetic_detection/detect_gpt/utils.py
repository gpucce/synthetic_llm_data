import re
import json
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import Dataset


PATTERN = re.compile(r"<extra_id_\d+>")

class PandasDataset(Dataset):
    def __init__(self, pd_data):
        self.df = pd_data

    def __getitem__(self, x):
        return self.df.iloc[x, :]

    def __len__(self):
        return self.df.shape[0]


def pandas_collate(x):
    return {i: [j[i] for j in x] for i in x[0].index}


def count_masks(texts):
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts
    ]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, model, tokenizer):
    n_expected = count_masks(texts)
    stop_id = tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(
        **{i:j.to("cuda") for i,j in tokens.items()},
        max_length=150,
        do_sample=True,
        top_p=1.0,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)


def tokenize_and_mask(text, span_length=2, pct=0.3, buffer_size=1, ceil_pct=False):
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    count = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1
        count += 1
        if count >= 1000:
            break

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = " ".join(tokens)
    return text


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [PATTERN.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]
    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(" ") for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def process_spaces(text):

    text = (
        text.replace(" ,", ",")
        .replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ;", ";")
        .replace(" '", "'")
        .replace(" ’ ", "'")
        .replace(" :", ":")
        .replace("<newline>", "\n")
        .replace("`` ", '"')
        .replace(" ''", '"')
        .replace("''", '"')
        .replace(".. ", "... ")
        .replace(" )", ")")
        .replace("( ", "(")
        .replace(" n't", "n't")
        .replace(" i ", " I ")
        .replace(" i'", " I'")
        .replace("\\'", "'")
        .replace("\n ", "\n")
        .strip()
    )

    return text


def _count_file_lines(data_path):
    with open(data_path) as f:
        nlines = sum(1 for i in f)
    return nlines

class CustomTrainDataLoader:
    def __init__(
        self, tokenizer, data_path, n_epochs, max_seq_len=512, batch_size=1, max_samples=None,
    ):
        self.tokenizer = tokenizer
        self.data_path = Path(data_path)
        self.n_epochs = n_epochs
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.max_samples = max_samples

        self.data_files = self.data_path.iterdir()
        data_path = next(self.data_files)
        self._n_samples = _count_file_lines(data_path)
        self.data = open(data_path)

        self.global_samples_done = 0
        self.samples_done = 0
        self.epochs_done = 0


    def __iter__(self):
        return self

    def _reopen(self):
        self.close()
        self.epochs_done += 1
        print(self.epochs_done)
        if self.epochs_done >= self.n_epochs:
            raise StopIteration

        try:
            data_path = next(self.data_files)
        except StopIteration as e:
            print(e)
            self.data_files = self.data_path.iterdir()
            data_path = next(self.data_files)

        self._n_samples = _count_file_lines(data_path)
        self.data = open(data_path)
        self.samples_done = 0

    def __next__(self):
        new_batch = []
        while True:
            if self.samples_done == self._n_samples:
                self._reopen()

            if len(new_batch) == self.batch_size:
                break

            new_sample = json.loads(next(self.data))
            self.global_samples_done += 1
            self.samples_done += 1

            if self.max_samples is not None and self.global_samples_done >= self.max_samples:
                raise StopIteration

            if len(new_sample) == self.max_seq_len:
                new_batch.append(new_sample)
            else:
                continue

        return torch.tensor(new_batch, dtype=int)

    def close(self):
        self.data.close()


def custom_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--accum-freq", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--col-name", type=str, default=None)
    parser.add_argument("--col-names", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--do-not-cache", action="store_false", default=True)
    parser.add_argument("--do-lora", action="store_true", default=False)
    parser.add_argument("--do-f16", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--is-hf", action="store_true", default=False)
    parser.add_argument("--is-torchrun", action="store_true", default=False)
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--hf-sample-top-p", default=False, action="store_true")
    parser.add_argument(
        "--modifier-model", type=str, default="t5-3b"
    )
    parser.add_argument("--n-samples", type=int, default=-1)
    parser.add_argument("--n-modifications", type=int, default=5)
    parser.add_argument("--optimizer-name", type=str, default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--output-path", type=str, default="test_output")
    parser.add_argument("--pct-mask", type=float, default=0.3)
    parser.add_argument("--keep-empty-lines", action="store_true", default=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--skip-epoch", type=int, default=0)
    parser.add_argument("--steps-per-ckpt", type=int, default=10000)
    parser.add_argument("--strict-load", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--wandb-project-name", type=str, default="llama_fine_tune")
    parser.add_argument("--warmup-ratio", type=float, default=0.01)
    return parser.parse_args()
