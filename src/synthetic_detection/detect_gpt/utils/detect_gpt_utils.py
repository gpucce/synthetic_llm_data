

import numpy as np
import torch
from torch.nn import CrossEntropyLoss


def compute_loss(texts, model, tokenizer):
    tokens = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**{i:j.to(model.device) for i,j in tokens.items()})
        labels = tokens["input_ids"].clone()
        pad_mask = labels == tokenizer.pad_token_id
        labels[pad_mask] = -100
        t_logits = outputs.logits.transpose(-1, -2)
        loss = per_sequence_crossentropy(t_logits, labels.to(t_logits.device))
        return loss.detach().cpu().tolist()

def per_sequence_crossentropy(inputs, labels):
    loss_fn = CrossEntropyLoss(reduction="none")
    return loss_fn(inputs, labels).mean(-1)

def compute_prob(x, axis=None):
    return np.sum(np.log(x), axis=axis) / len(x)


def detect_gpt(df, true_col):
    df_true = df.loc[:, true_col].apply(compute_prob).values
    df_syn = (
        df.loc[:, [i for i in df.columns if i != true_col]].applymap(compute_prob).values
    )

    mean_df_syn = np.mean(df_syn, axis=1)
    unnorm_denom = df_syn - mean_df_syn.reshape(-1, 1)
    denom = np.sqrt(np.power(unnorm_denom, 2).sum(axis=1) / (df_syn.shape[1] - 1))
    return (df_true - mean_df_syn) / denom
