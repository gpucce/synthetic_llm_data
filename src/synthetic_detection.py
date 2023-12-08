"""This file is used to train a model to detect whether a text is human or machine generated."""

import os
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from data_handling.utils import read_jsonl, read_many_jsonl, parse_args



def main(args):
    """main function"""
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    try:
        _dataset = read_jsonl(args.data_path)
    except:
        _dataset = read_many_jsonl(args.data_path)

    train_idx, test_idx = train_test_split(
        range(len(_dataset)),
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    _train_dataset = [_dataset[i] for i in train_idx]
    _test_dataset = [_dataset[i] for i in test_idx]
    def create_dataset(_dataset):
        return Dataset.from_dict(
            {
                "text": [i["human_text"] for i in _dataset]
                + [i["machine_text"] for i in _dataset],
                "label": [0 for _ in _dataset] + [1 for _ in _dataset],
            }
        ).map(
            lambda x: tokenizer(
                x["text"], padding=True, truncation=True, max_length=512
            ),
            batched=True
        )

    ds = DatasetDict(
        {
            "train":create_dataset(_train_dataset),
            "test":create_dataset(_test_dataset)
        }
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path)
    output_dir = os.path.join(
        "detection_tests",
        args.model_name_or_path.split("/")[-1],
        os.path.basename(args.data_path).replace(".jsonl", ""),
        str(args.seed),
    )
    print("OUTPUT DIR:", output_dir)
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=64,
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
        learning_rate=1e-5,
        seed=args.seed,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        return {
            "accuracy":(predictions.argmax(1) == labels).sum() / len(labels)
        }

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

    print(results)


if __name__ == "__main__":
    main(parse_args())
