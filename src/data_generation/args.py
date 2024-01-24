
from argparse import ArgumentParser
from ..utils import str2bool

def get_base_parser():
    parser = ArgumentParser()

    parser.add_argument("--columns_to_remove", nargs="+", default=None)
    parser.add_argument('--eos_token_id', type=int, default=None)
    parser.add_argument("--huggingface_or_vllm", type=str,
        default="huggingface", choices=["huggingface", "vllm"])
    parser.add_argument("--human_key", type=str, default=None)
    parser.add_argument("--is_test", type=str2bool, default=False)
    parser.add_argument("--max_batch_size", type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=150)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--min_new_tokens", type=int, default=20)
    parser.add_argument("--n_few_shots", type=int, default=0)
    parser.add_argument("--name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument('--pad_token_id', type=int, default=None)
    parser.add_argument("--padding_side", type=str, default="left")
    parser.add_argument("--preprocessing", type=str, default=None)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument("--use_beam_search", type=str2bool, default="true")
    return parser

def parse_generate_args():
    """Parse commandline arguments."""
    parser = get_base_parser()
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--load_in_8bit", type=str2bool, default="false")
    parser.add_argument("--n_tasks", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--system_prompt", type=str, default=None)

    return parser.parse_args()


def parse_complete_args():
    parser = get_base_parser()
    parser.add_argument("--base_path", type=str, default=".")
    parser.add_argument("--project", type=str, default="semeval_task_3")
    parser.add_argument("--selected_boundary", type=int, default=None)
    parser.add_argument("--split_at_random_length", type=str2bool, default=True)
    parser.add_argument("--split_files", type=str, nargs="+", required=True)
    parser.add_argument("--split_names", type=str, nargs="+", default=True)

    args = parser.parse_args()
    assert len(args.split_files) == len(args.split_names)

    return args
