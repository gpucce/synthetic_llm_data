"""Complete a given text"""
import pdb
import json
import re
import random
from tqdm.auto import tqdm
# from nltk import sent_tokenize
import transformers as ts
import torch

from .utils import read_PeerRead

# old_prompt = """Task:
# Complete a partially-written peer review of the paper.

# Make sure that:
# 1. The completion is of at least {num_of_words} words.
# 2. You only output the completion of the partial review and not write it from the beginning.

# Title of the paper:
# {paper_title}

# Abstract of the paper:
# {paper_abstract}

# Partially-written review:
# \"\"\"
# {partial_review}
# \"\"\"

# Continuation:
# """

prompt = """Title of the paper:
{paper_title}

Abstract of the paper:
{paper_abstract}

Review:
{partial_review}
"""

def parse_completion(completion):
    return json.loads(completion)["completion"]

def split_by_full_stop(text):
    return re.split(r"(?<![A-Z\d])[.!?] +(?=[A-Z])", text)

def generate(text, model, tokenizer):
    ids = tokenizer(text, truncation=True, return_tensors='pt')
    out = model.generate(
        ids['input_ids'].to(model.device),
        # min_new_tokens=10,
        max_new_tokens=max(len(ids['input_ids']), 1024),
        return_dict_in_generate=True,
        output_scores=True,
        no_repeat_ngram_size=6,
        repetition_penalty=1.05,
    )
    return tokenizer.decode(out.sequences[0], True)

def get_prompt(data_item):
    """
    A part of human review could be cut off 
    1. randomly at a word present in 1/10th to 5/10th of the review.
    2. randomly at a closest sentence boundary in the 1/10th to 5/10th of the review.
    """

    human_reviews = data_item["reviews"]
    human_end_boundaries = []
    cut_at_sentences = []
    task3_prompts = []

    # select one out of all the human reviews available
    # human_review = human_reviews[random.randint(0, len(human_review) - 1)]
    for human_review in human_reviews:
        human_review = human_review["comments"]
        cut_at_sentence = random.randint(0, 2) > 1
        cut_at_sentences.append(cut_at_sentence)

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
        updated_prompt = prompt.format(
            paper_title=data_item["title"],
            paper_abstract=data_item["abstract"],
            partial_review=partial_review,
            num_of_words=num_of_words_to_generate,
        )
    
        task3_prompts.append(updated_prompt)
        human_end_boundaries.append(selected_boundary)

    data_item["human_end_boundaries"] = human_end_boundaries
    data_item["cut_at_sentences"] = cut_at_sentences
    data_item["task3_prompts"] = task3_prompts
    keys_to_pop = ["chatgpt_reviews", "davinci_reviews", "prompts"]
    for key in keys_to_pop:
        data_item.pop(key, None)

    assert len(data_item["reviews"]) == len(data_item["task3_prompts"])

    return data_item

    # return {
    #     "id": data_item["id"],
    #     "source": data_item["source"],
    #     "title": data_item["title"],
    #     "num_of_words_to_generate": num_of_words_to_generate,
    #     "human_end_boundary": selected_boundary,
    #     "cut_at_sentence": cut_at_sentence,
    #     # "partial_review": partial_review,
    #     "task3_prompt": updated_prompt,
    #     "human_review": human_review,
    #     "partial_review": partial_review,
    # }


print("Loading the data...")

data = read_PeerRead("/leonardo_scratch/large/userexternal/gpuccett/data/PeerRead/")
data_keys = list(data[0].keys())

print('Loading the model...')
# checkpoint = '/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/tiny-random-llama'
checkpoint = '/leonardo_scratch/large/userexternal/gpuccett/models/hf_llama/llama-2-7b-hf'

model = ts.AutoModelForCausalLM.from_pretrained(
    checkpoint,torch_dtype=torch.bfloat16,device_map='auto',)

tokenizer = ts.AutoTokenizer.from_pretrained(checkpoint)
print('Model successfully loaded.')


outputs = []

for data_item in data:
    data_item = data_item["data"]
    item = get_prompt(data_item)
    task3_prompts = item["task3_prompts"]
    item["machine_review"] = []
    for task3_prompt in tqdm(task3_prompts):
        # try:
        print(task3_prompt)
        print('Generating...')
        complete_review = generate(task3_prompt, model, tokenizer)
        # print(f"\n\n{'='*50}\nReview:\n", complete_review, f"\n{'='*50}\n")
        item["machine_review"].append(complete_review)
    outputs.append(item)
    with open('outputs.json', 'w') as f:
        json.dump(outputs, f)

        # except Exception as e:
        #     print(e)
        #     pdb.set_trace()
        #     item["machine_review"].append("")
        #     continue

    with open("PeerRead_review_completions_final.jsonl", "a") as f:
        f.write(json.dumps(item) + "\n")

    outputs.append(item)

with open("PeerRead_review_completions_final.json", "w") as f:
    json.dump(outputs, f, indent=4)
