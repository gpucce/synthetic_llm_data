import re
import random
import numpy as np

PATTERN = re.compile(r"<extra_id_\d+>")


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")])
            for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts, model, tokenizer, args):
    n_expected = count_masks(texts)
    stop_id = tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    outputs = model.generate(
        **{i:j.to(model.device) for i,j in tokens.items()},
        do_sample=True,
        top_p=1.0,
        num_return_sequences=1,
        eos_token_id=stop_id,
        max_new_tokens=150
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=False)


def _replace_mask_string(tokens, mask_string):
    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    return tokens, num_filled

def tokenize_and_mask_with_indices(text, start, stop):
    mask_string = "<<<mask>>>"
    text = text[:start] + f" {mask_string} " + text[stop + 1:] # + 1 because "end" is not 0-indexed
    text = re.sub(" +", " ", text)
    tokens = text.split(" ")
    tokens, _ = _replace_mask_string(tokens, mask_string)
    text = " ".join(tokens)
    return text

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

    tokens, num_filled = _replace_mask_string(tokens, mask_string)
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
    return [process_spaces(i) for i in texts]


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


def multiple_length_cutting(x, col_name, length):
    max_length = length // 10
    interval = random.randint(max_length - 15, max_length)
    return {col_name: " ".join(x[col_name].split(" ")[:length - interval * 10])}