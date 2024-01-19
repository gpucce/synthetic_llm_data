
import re
import numpy as np
import torch
from torch.nn import CrossEntropyLoss


def compute_loss(texts, model, tokenizer, loss_type="non_multilabel"):
    tokens = tokenizer(
        texts, return_tensors="pt", padding=True,
        truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**{i:j.to(model.device) for i,j in tokens.items()})
        labels = tokens["input_ids"].clone()
        pad_mask = labels == tokenizer.pad_token_id
        labels[pad_mask] = -100
        logits = outputs.logits[..., :-1, :]
        labels = labels[..., 1:]
        if loss_type == "multilabel":
            t_logits = logits.transpose(-1, -2)
            loss = multilabel_per_sequence_crossentropy(
                t_logits, labels.to(t_logits.device)).detach().cpu().tolist()
        else:
            loss = per_sequence_crossentropy(logits, labels.to(logits.device))
        return loss

def compute_loglikelihood(texts, model, tokenizer):
    return [-i for i in compute_loss(texts, model, tokenizer)]

def per_sequence_crossentropy(inputs, labels):
    loss_fn = CrossEntropyLoss(ignore_index=-100)
    out = []
    for lo, la in zip(inputs, labels):
        out.append(loss_fn(lo.view(-1, lo.size(-1)), la.view(-1)).detach().cpu().item())
    return out

def multilabel_per_sequence_crossentropy(inputs, labels):
    loss_fn = CrossEntropyLoss(reduction="none")
    return loss_fn(inputs, labels).mean(-1)

def compute_prob(x, axis=None):
    return np.sum(np.log(x), axis=axis) / len(x)

def batched_detect_score(x, col_name="document_loss", re_pattern="synthetic.*loss"):
    _array = np.array([x[i] for i in x if re.search(re_pattern, i)])
    return (np.array(x[col_name]) - np.mean(_array, axis=0)) / np.std(_array, axis=0)
