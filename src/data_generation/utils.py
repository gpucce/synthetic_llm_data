
import re
import ast
import random

NON_CHAT_CONTINUATION = """Title of the paper:
{paper_title}

Abstract of the paper:
{paper_abstract}

Review:
{partial_review}"""

CHAT_CONTINUATION="""Task:
Complete a partially-written peer review of the following paper.

Make sure that:
1. The completion is of at least {num_of_words} words.
2. You only complete the partial review and not write it from the beginning.

Title of the paper:
{paper_title}

Abstract of the paper:
{paper_abstract}

Review:
{partial_review}"""

PROMPT_REGISTRY = {
    "semeval_task_3": {
    **{model_name:NON_CHAT_CONTINUATION
       for model_name in ["llama-2-7b-hf", "llama-2-13b-hf", "llama-2-70b-hf"]},
    **{model_name:CHAT_CONTINUATION
       for model_name in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]}
    }
}

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

def get_semeval_task3_prompt(data_item, model_name):
    """
    A part of human review could be cut off 
    1. randomly at a word present in 1/10th to 5/10th of the review.
    2. randomly at a closest sentence boundary in the 1/10th to 5/10th of the review.
    """


    human_review = (data_item["full_human_review"]
        if "full_human_review" in data_item
        else data_item["human_review"])

    cut_at_sentence = random.randint(0, 2) > 1

    sentences = split_by_full_stop(human_review)
    words = human_review.split(" ")

    num_of_words = len(words)
    sentence_boundaries = [0] + [len(sentence.split(" ")) for sentence in sentences]
    sentence_boundaries = [
        sum(sentence_boundaries[:i]) for i in range(1, len(sentence_boundaries))
    ]

    # selecting human review from 1/10th to the 5/10th of the review
    selected_human_review_word = random.randint(
        int(num_of_words / 10), int(num_of_words / 2)
    )
    selected_human_review = words[:selected_human_review_word]

    if cut_at_sentence:
        # get the closest sentence boundary
        distance_to_sentence_boundary = [
            abs(len(selected_human_review) - boundary)
            for boundary in sentence_boundaries
        ]
        selected_boundary = sentence_boundaries[
            distance_to_sentence_boundary.index(min(distance_to_sentence_boundary))
        ]
    else:
        selected_boundary = selected_human_review_word

    num_of_words_to_generate = num_of_words - selected_boundary
    partial_review = words[:selected_boundary]
    partial_review = " ".join(partial_review)

    updated_prompt = PROMPT_REGISTRY["semeval_task_3"][model_name].format(
        paper_title=data_item["title"],
        paper_abstract=data_item["abstract"],
        partial_review=partial_review,
        num_of_words=num_of_words_to_generate,
    )

    data_item["full_human_review"] = human_review
    data_item["cut_at_sentence"] = cut_at_sentence
    data_item["human_end_boundary"] = selected_boundary
    data_item["prompt"] = updated_prompt
    data_item["truncated_human_review"] = partial_review

    return data_item

def add_newline(x):
    return x + "\n" if x[-1] != "\n" else x

def preprocessing_bloomz(prompts):
    return [add_newline(prompt.replace("325 words", "1000 words"))
            for prompt in prompts]

def preprocessing_gpt4(prompts):
    if not isinstance(prompts, list):
        prompts = ast.literal_eval(prompts)
    return " ".join(prompts)


def get_preprocessing_func(args):
    if args.preprocessing == "bloomz":
        return preprocessing_bloomz
    return lambda x: x

def save_to_right_format(ds, output_file):
    output_file = str(output_file)
    if output_file.endswith(".jsonl"):
        ds.to_json(output_file, orient="records", lines=True)
    elif output_file.endswith(".json"):
        ds.to_json(output_file, lines=False)
    elif output_file.endswith(".csv"):
        ds.to_csv(output_file)
    else:
        raise ValueError(f"Unknown output file format {output_file}")
