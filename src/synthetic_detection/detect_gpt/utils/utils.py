
from argparse import ArgumentParser

def custom_parse_args():
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--col-names", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--model-name", type=str, default="llama-7b-chat")
    parser.add_argument("--modifier-model", type=str, default="t5-3b")
    parser.add_argument("--n-modifications", type=int, default=5)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--output-path", type=str, default="test_output")
    parser.add_argument("--pct-mask", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()
