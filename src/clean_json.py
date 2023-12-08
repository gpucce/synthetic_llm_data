"""postprocess generation json file"""
import json
import re
from argparse import ArgumentParser
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from langdetect import detect
from .utils import read_many_jsonl

_CLEAN_RECORD_FUNCS = {}

def _camel2snake(s):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

def register_record_cleaner(cls):
    """Decorator registering list cleaners"""
    _CLEAN_RECORD_FUNCS[_camel2snake(cls.__name__)] = cls
    return cls


@register_record_cleaner
def _clean_newlines(text):
    return text.replace("\n", " ").strip()
    

@register_record_cleaner
def _clean_given_string(text):
    if "#" in text:
        text = text.split("#")[0].strip()
    if "Home" in text:
        text = text.split("Home")[0].strip()
    return text
    
@register_record_cleaner
def _clean_whitespace(text):
    return re.sub(" +", " ", text)

@register_record_cleaner
def _cut_with_tokenizer(text, tokenizer):
    return tokenizer.decode(
        tokenizer(text, add_special_tokens=False).input_ids[:400])

@register_record_cleaner
def _split_on_dots(text):
    return ".".join(text.split(".")[:-1])

def postprocess_record(sample, tokenizer, _keys):
    """postprocess function"""
    out = {}
    for _key, _val in sample.items():
        out[_key] = _val
        if _key not in _keys:
            continue

        for fname, func in _CLEAN_RECORD_FUNCS.items():
            if "tokenizer" in fname:
                out[_key] = func(out[_key], tokenizer)
            else:
                out[_key] = func(out[_key])

    return out

_CLEAN_LIST_FUNCS = {}

def register_list_cleaner(cls):
    """Decorator registering record cleaners"""
    _CLEAN_LIST_FUNCS[_camel2snake(cls.__name__)] = cls
    return cls

@register_list_cleaner
def _remove_too_short(text):
    return len(text) > 1000

@register_list_cleaner
def _remove_non_ita(text):
    try:
        out = detect(text[-300:]) == "it" and detect(text[-100:]) == "it"
    except:
        print(text)
        return False
    return out

def postprocess_list(text_list, tokenizer, _keys):
    for fname, func in _CLEAN_LIST_FUNCS.items():
        if "tokenizer" in fname:
            text_list = [sample for sample in text_list if all([func(text, tokenizer) for _key, text in sample.items() if _key in _keys])]
        else:
            text_list = [sample for sample in text_list if all([func(text) for _key, text in sample.items() if _key in _keys])]

    return text_list

def parser_args():
    """local parser"""
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--max_samples", type=int, default=-1)
    return parser.parse_args()

def main(args):
    """main function"""
    print("RECORD_CLEAN_FUNCS", _CLEAN_RECORD_FUNCS)
    print("LIST_CLEAN_FUNCS", _CLEAN_LIST_FUNCS)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, fast=True)
    input_generation = read_many_jsonl(args.input_path)

    _keys = ["human_text", "machine_text"]
    
    input_generation = postprocess_list(
        [postprocess_record(sample, tokenizer, _keys=_keys) for sample in input_generation], 
        tokenizer, _keys=_keys)

    if args.max_samples > 0:
        input_generation = input_generation[:args.max_samples]

    with open(args.output_path, "w") as jfw:
        for i in input_generation:
            jfw.write(json.dumps(i))
            jfw.write("\n")

if __name__ == "__main__":
    main(parser_args())
