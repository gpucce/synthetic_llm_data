
from argparse import ArgumentParser

def _parse_args():
    parser = ArgumentParser()
    parser.add_argument("--col-name", type=str, default="text")
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--name-or-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser

def lmrater_parse_args():
    parser = _parse_args()
    parser.add_argument("--output-path", type=str)
    parser.add_argument("--preprocessing", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()

def replacement_parse_args():
    parser = _parse_args()
    parser.add_argument("--n-modifications", type=int)
    parser.add_argument("--preprocessing", type=str)
    return parser
