"""This file is used to train a model to detect whether a text is human or machine generated."""

import os
import json
from pathlib import Path
from argparse import ArgumentParser
from datasets import load_from_disk, disable_caching, Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve
from ..data_generation.prompts import PromptPreprocessor

disable_caching()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--col_name", type=str, help="Name of column to use")
    parser.add_argument("--data_path", type=str, help="Path to dataset")
    parser.add_argument("--real_data_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=4000)
    parser.add_argument("--output_path", type=str, help="Path to save the model")
    parser.add_argument("--model_name_or_path", type=str, help="Model name or path")
    parser.add_argument("--seed", type=int, help="Random seed", default=[42])
    parser.add_argument("--lr", type=float, default=1e-5)
    return parser.parse_args()


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    _dataset = load_from_disk(args.data_path)

    train_idx, test_idx = train_test_split(
        range(len(_dataset)),
        train_size=args.max_samples * 2,
        test_size=args.max_samples // 2,
        random_state=42,
        shuffle=True,
        stratify=_dataset["label"]
    )

    ds = DatasetDict(
        {
            "train":_dataset.select(train_idx),
            "test":_dataset.select(test_idx)
        }
    )

    assert len(ds["train"]) == args.max_samples * 2, \
        f"{len(ds['train'])} != {args.max_samples * 2}"

    if args.real_data_path is not None:
        args.preprocessing = "change_it"
        args.name_or_path = args.data_path.split("/")[-2]
        args.human_key = args.col_name
        args.project = "ita_news"
        args.split_at_random_length = False
        args.selected_boundary = 60
        args.n_few_shots = 0

        real_dataset = load_dataset(
            args.real_data_path.split(".")[-1], data_files=args.real_data_path)["train"]
        preprocessing_fn = PromptPreprocessor(args, real_dataset)
        real_dataset = real_dataset.map(
            preprocessing_fn, desc="Generating prompts").shuffle(seed=args.seed)

        _human_dataset = ds["train"].filter(lambda x: x["label"] == 0).shuffle(seed=args.seed)
        _synth_dataset = ds["train"].filter(lambda x: x["label"] == 1).shuffle(seed=args.seed)

        assert len(_human_dataset) == len(_synth_dataset), \
            f"{len(_human_dataset)} != {len(_synth_dataset)}"

        split_n_samples = min(len(_human_dataset) // 2, args.max_samples // 2)

        _dataset = Dataset.from_dict(
            {
                args.col_name: (
                    _synth_dataset[args.col_name][: 2 * split_n_samples] +
                    _human_dataset[args.col_name][:split_n_samples] +
                    real_dataset["human_text"][:split_n_samples]),
                "label": (
                    _synth_dataset["label"][: 2 * split_n_samples] +
                    _human_dataset["label"][:split_n_samples] +
                    [0] * split_n_samples)
            }
        )

        _dataset = _dataset.map(lambda x:
            {args.col_name: " ".join(x[args.col_name].split(" ")[:150])})

        ds["train"] = _dataset

    output_dir = os.path.join(args.output_path + f"_seed_{args.seed}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_dir, "args.txt"), "w") as f:
        json.dump(vars(args), f)

    ds_human = ds["train"].filter(lambda x: x["label"] == 0)
    ds_synth = ds["train"].filter(lambda x: x["label"] == 1)
    for i in range(10):
        print(ds_human[i][args.col_name])
        print(ds_synth[i][args.col_name])

    for seed in args.seed:
        ds = ds.shuffle(seed=args.seed)
        ds = ds.map(
            lambda x: tokenizer(
                x[args.col_name],
                padding="max_length",
                truncation=True,
                max_length=512),
            batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, num_labels=2)

        print("OUTPUT DIR:", output_dir)
        training_args = TrainingArguments(
            output_dir=output_dir + f"_seed_{seed}",
            do_train=True,
            do_eval=True,
            per_device_train_batch_size=32,
            evaluation_strategy="steps",
            eval_steps=100,
            per_device_eval_batch_size=64,
            num_train_epochs=3,
            warmup_steps=0,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            metric_for_best_model="accuracy",
            overwrite_output_dir=True,
            learning_rate=args.lr,
            seed=seed,
        )

        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            out = {"accuracy":(predictions.argmax(1) == labels).sum() / len(labels)}
            fpr, tpr, _ = roc_curve(labels, predictions[:, 1])
            out["auroc"] = auc(fpr, tpr)
            return out
    
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        results = trainer.evaluate()
        trainer.save_model()
        trainer.save_state()
        ds.save_to_disk(os.path.join(output_dir, "dataset"))

        print(results)


if __name__ == "__main__":
    main(parse_args())
