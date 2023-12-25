"""Utility functions for data handling"""

import os
import re
import json
import time
import shutil
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
import datasets


def split_by_full_stop(text):
    return re.split(r"(?<![A-Z\d])[.!?] +(?=[A-Z])", text)

def read_jsonl(file_path):
    """Read a jsonl file into a list of dicts"""
    with open(file_path, encoding="utf-8") as jf:
        lines = [json.loads(i) for i in jf.readlines()]
    return lines

def read_many_jsonl(dir_path):
    jsonls = [os.path.join(dir_path, i) for i in os.listdir(dir_path)
                if i.endswith("jsonl") or i.endswith("json")]
    jsonls.sort()
    jsonls = [read_jsonl(i) for i in jsonls]
    lines = []
    for i in jsonls:
        lines += i
    return lines


def read_PeerRead(dir_path):
    path = Path(dir_path)
    all_files = []
    progressbar = tqdm()
    for conf_data in (path / "data").iterdir():
        if not conf_data.is_dir():
            continue
        print(conf_data)
        for train_test_dev in conf_data.iterdir():
            if train_test_dev.name not in ["train", "test", "dev"]:
                continue
            for file in (train_test_dev / "reviews").iterdir():

                with open(file, "r") as jf:
                    current_file = json.load(jf)
                for comment in current_file["reviews"]:
                    all_files.append({
                        "conf":conf_data.name, 
                        "file":file.name,
                        "train_test_dev":train_test_dev.name,
                        "title": current_file["title"],
                        "abstract": current_file["abstract"],
                        "comment": comment["comments"],                        
                    })

                    progressbar.update()
    ds = datasets.Dataset.from_list(all_files)
    return ds

def parse_args():
    """Parse command line arguments"""
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    return v.lower() in ("yes", "true", "t", "1")

def save_distributed_and_collect_on_main_rank(
        data_shard, global_rank, global_n_devices,
        output_file, save_after_collect=True
    ):

    if global_n_devices == 1:
        data_shard.save_to_disk(output_file)
        return data_shard

    file_name_template = "{output_file}_n_shards_{global_n_devices}_shard_id_{shard_id}"
    local_file_name = file_name_template.format(
        output_file=output_file,
        global_n_devices=global_n_devices,
        shard_id=global_rank
    )

    data_shard.save_to_disk(local_file_name)
    # molto fatto a mano ma sembra funzionare
    all_files = [
        file_name_template.format(
            output_file=output_file,
            global_n_devices=global_n_devices,
            shard_id=i
        )
        for i in range(global_n_devices)
    ]

    if global_rank == 0:

        all_data_there = False
        while not all_data_there:
            time.sleep(2)
            all_data_there = all(
                [(Path(file_name) / "dataset_info.json").exists() for file_name in all_files])

        dataset = datasets.concatenate_datasets([
                datasets.load_from_disk(file_name) for file_name in all_files])

        if save_after_collect:
            dataset.save_to_disk(output_file)
        for i in range(global_n_devices):
            shutil.rmtree(f"{output_file}_n_shards_{global_n_devices}_shard_id_{i}")
        return dataset
