"""Translate datasets using a trained model"""
from argparse import ArgumentParser
import os
import time
import shutil
import datasets
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def parse_args():
    """Parse command-line arguments"""
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--max_samples', type=int, default=-1)
    return parser.parse_args()


class SlimOrcaTranslator:
    """Model wrapper for translating slim_orca"""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tok = tokenizer

    def generate(self, batch, device_id):
        """Generate a translation for a batch of text"""
        return self.model.generate(
                self.tok(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).input_ids.to(device_id),
                tgt_lang="ita"
            ).detach().to("cpu")

    def __call__(self, batch, device_id):
        """Generate a translation for a slim_orca example"""
        out = {"conversations": []}
        self.model.to(device_id)

        with torch.no_grad():
            system = [i[0]["value"] for i in batch["conversations"]]
            system = self.tok.batch_decode(
                self.generate(system, device_id),
                skip_special_tokens=True)

            user = [i[1]["value"] for i in batch["conversations"]]
            user = self.tok.batch_decode(
                self.generate(user, device_id),
                skip_special_tokens=True)

            gpt = [i[2]["value"] for i in batch["conversations"]]
            gpt = self.tok.batch_decode(
                self.generate(gpt, device_id),
                skip_special_tokens=True)

        for s, u, g in zip(system, user, gpt):
            out["conversations"].append([
                {"from": "system", "value":s},
                {"from": "user", "value":u},
                {"from": "gpt", "value":g}
            ])

        return out

def get_translator(dataset_type, model, tokenizer):
    """Get the generate function for a dataset"""
    if dataset_type == "slim_orca_dedup":
        return SlimOrcaTranslator(model, tokenizer)

def main(args):
    """Main function"""
    dataset = datasets.load_from_disk(args.input_file)

    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    global_n_devices = int(os.environ.get("SLURM_NTASKS", 1))
    print(f"global rank: {global_rank}, local rank: {local_rank}, n_devices: {global_n_devices}")

    if "train" in dataset:
        dataset = dataset["train"]

    if args.max_samples > 0:
        dataset = dataset.select(range(args.max_samples))

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name, low_cpu_mem_usage=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    translator = SlimOrcaTranslator(model, tokenizer)

    dataset = dataset.shard(num_shards=global_n_devices, index=global_rank).map(
        lambda batch: translator(batch, local_rank),
        batched=True,
        batch_size=64
    )

    dataset.save_to_disk(f"{args.output_file}_n_shards_{global_n_devices}_shard_id_{global_rank}")

    # molto fatto a mano ma sembra funzionare
    if global_rank == 0:

        all_data_there = False
        while not all_data_there:
            time.sleep(2)
            all_data_there = all(
                [
                    os.path.exists(
                        f"{args.output_file}_n_shards_{global_n_devices}_shard_id_{i}/dataset_info.json")
                    for i in range(global_n_devices)
                ]
            )

        dataset = datasets.concatenate_datasets([
                datasets.load_from_disk(
                    f"{args.output_file}_n_shards_{global_n_devices}_shard_id_{i}")
                for i in range(global_n_devices)
            ])
        dataset.save_to_disk(args.output_file)
        for i in range(global_n_devices):
            shutil.rmtree(f"{args.output_file}_n_shards_{global_n_devices}_shard_id_{i}")

if __name__ == "__main__":
    main(parse_args())
