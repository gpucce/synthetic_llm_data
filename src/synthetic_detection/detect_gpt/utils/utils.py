
import json
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset

def custom_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--col-names", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--modifier-model", type=str, default="t5-3b")
    parser.add_argument("--n-modifications", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=-1)
    parser.add_argument("--output-path", type=str, default="test_output")
    parser.add_argument("--pct-mask", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


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

def _count_file_lines(data_path):
    with open(data_path) as f:
        nlines = sum(1 for i in f)
    return nlines

class PandasDataset(Dataset):
    def __init__(self, pd_data):
        self.df = pd_data

    def __getitem__(self, x):
        return self.df.iloc[x, :]

    def __len__(self):
        return self.df.shape[0]

def pandas_collate(x):
    return {i: [j[i] for j in x] for i in x[0].index}
