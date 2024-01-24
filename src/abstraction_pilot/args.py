

from synthetic_llm_data.src.data_generation.args import get_base_parser

def lmrater_parse_args():
    parser = get_base_parser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--project", type=str, default="wemb")
    parser.add_argument("--selected_boundary", type=int, default=None)
    parser.add_argument("--split_at_random_length", action="store_true", default=False)
    return parser.parse_args()

def replacement_parse_args():
    parser = get_base_parser()
    parser.add_argument("--col_name", type=str, default="text")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--n_modifications", type=int)
    return parser.parse_args()
