"""Utility functions for data handling"""

import os
import json
from pathlib import Path
from argparse import ArgumentParser


def read_jsonl(file_path):
    """Read a jsonl file into a list of dicts"""
    with open(file_path, encoding="utf-8") as jf:
        lines = [json.loads(i) for i in jf.readlines()]
    return lines

def read_many_jsonl(dir_path):
    jsonls = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith("jsonl") or i.endswith("json")]
    jsonls.sort()
    jsonls = [read_jsonl(i) for i in jsonls]
    lines = []
    for i in jsonls:
        lines += i
    return lines


def read_PeerRead(dir_path):
    path = Path(dir_path)
    all_files = []
    for conf_data in (path / "data").iterdir():
        if "arxiv" in str(conf_data) or not conf_data.is_dir():
            continue
        for train_test_dev in conf_data.iterdir():
            if train_test_dev.name not in ["train", "test", "dev"]:
                continue
            for file in (train_test_dev / "reviews").iterdir():
                with open(file, "r") as jf:
                    current_file = json.load(jf)
                    all_files.append({
                        "conf":conf_data.name, 
                        "file":file.name,
                        "train_test_dev":train_test_dev.name,
                        "data":current_file
                    })
    return all_files

def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()