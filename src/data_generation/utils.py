import re
import random

def camoscio_preprocessing_function(inp):
    """Format the text string."""

    ITA_PROMPT_FORMAT_NO_INPUT = "Di seguito è riportata una istruzione che descrive un task. " + \
        "Scrivete una risposta che completi in modo appropriato la richiesta.\n\n" + \
        "### Istruzione:\n{instruction}\n\n### Risposta:\n"
    ITA_PROMPT_FORMAT = "Di seguito è riportata una istruzione che descrive un task. " + \
        "Scrivete una risposta che completi in modo appropriato la richiesta.\n\n" + \
        "### Istruzione:\n{instruction}\n\n### Input:\n{input}\n\n### Risposta:\n"

    try:
        if inp["input"] != "":
            prompt = ITA_PROMPT_FORMAT.format(
                instruction=inp["instruction"], input=inp["input"]
            )
        else:
            prompt = ITA_PROMPT_FORMAT_NO_INPUT.format(
                instruction=inp["instruction"])
        response = inp["output"]

    except Exception as e:
        raise ValueError(
            f"Unable to extract prompt/response from {inp=}") from e

    prompt = prompt + response
    return {"text": prompt}


def split_by_full_stop(text):
    return re.split(r"(?<![A-Z\d])[.!?] +(?=[A-Z])", text)

def add_newline(x):
    return x + "\n" if x[-1] != "\n" else x


def preprocessing_bloomz_peerread(prompt_dict):
    prompts = prompt_dict["prompt"]
    return [add_newline(prompt.replace("325 words", "1000 words"))
        for prompt in prompts]

def get_preprocessing_func(args):
    if args.preprocessing == "bloomz_peerread":
        return preprocessing_bloomz_peerread
    elif args.preprocessing == "camoscio":
        return camoscio_preprocessing_function

    return lambda x: x["prompt"]

def get_min_length_logits_processor(min_length, eos_token_id):
    def min_length_logits_processor(seq_ids, logits_row):
        true_min_length = random.randint(max(min_length - 200, 0), min_length)
        if len(seq_ids) < true_min_length:
            logits_row[eos_token_id] = float(-1e4)
        return logits_row
    return min_length_logits_processor

def get_strict_min_length_logits_processor(min_length, eos_token_id):
    def min_length_logits_processor(seq_ids, logits_row):
        if len(seq_ids) < min_length:
            logits_row[eos_token_id] = float(-1e4)
        return logits_row
    return min_length_logits_processor
