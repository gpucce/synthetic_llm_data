import random
from pathlib import Path

from .utils import split_by_full_stop

PEERREAD_NON_CHAT_CONTINUATION="""Title of the paper:
{paper_title}

Abstract of the paper:
{paper_abstract}

Review:
{partial_review}"""

PEERREAD_CHAT_CONTINUATION="""Task:
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

OUTFOX_CHAT_CONTINUATION = """Act as an experienced essay writer.

Given the following Problem statement (essay topic) complete the following Partial Essay writing at least 318 words with a clear opinion. The written essay should look like human-written (please start writing the essay without any additional text).

Problem statement (essay topic):
{essay_topic}

Partial Essay:
{partial_essay}"""

PROMPT_REGISTRY = {
    "semeval_task_3" : { 
        "peerread" : {
            **{model_name:PEERREAD_NON_CHAT_CONTINUATION for model_name
               in ["llama-2-7b-hf", "llama-2-13b-hf", "llama-2-70b-hf", "gpt2-hf"]},
            **{model_name:PEERREAD_CHAT_CONTINUATION for model_name
               in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf", "gpt2-hf"]}
        },
        "outfox" : {
            model_name:OUTFOX_CHAT_CONTINUATION for model_name
                in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf", "gpt2-hf"]
        }
    }
}

class PromptPreprocessor():
    def __init__(self, args):
        self.preprocessing = args.preprocessing
        self.model_name = Path(args.name_or_path).name
        self.human_key = args.human_key
        self.prompt = PROMPT_REGISTRY["semeval_task_3"][self.preprocessing][self.model_name]

    def interpolate_peerread_prompt(
        self, data_item, partial_prompt, num_of_words_to_generate, **kwargs):
        
        return self.prompt.format(
            paper_title=data_item["title"],
            paper_abstract=data_item["abstract"],
            partial_review=partial_prompt,
            num_of_words=num_of_words_to_generate,
        )

    def interpolate_outfox_prompt(self, data_item, partial_prompt, **kwargs):
        return self.prompt.format(
            essay_topic=data_item["topic"],
            partial_essay=partial_prompt
        )

    def interpolate_prompt(self, data_item, partial_prompt, **kwargs):
        if self.preprocessing == "peerread":
            return self.interpolate_peerread_prompt(
                data_item, partial_prompt, **kwargs)
        elif self.preprocessing == "outfox":
            return self.interpolate_outfox_prompt(
                data_item, partial_prompt, **kwargs)

        raise ValueError(f"Unknown formatting {self.formatting}")


    def __call__(self, data_item, **kwargs):
        """
        A part of human review could be cut off 
        1. randomly at a word present in 1/10th to 5/10th of the review.
        2. randomly at a closest sentence boundary in the 1/10th to 5/10th of the review.
        """

        human_review = data_item[self.human_key]

        cut_at_sentence = random.randint(0, 2) > 1

        sentences = split_by_full_stop(human_review)
        words = human_review.split(" ")

        num_of_words = len(words)
        sentence_boundaries = [0] + [len(sentence.split(" ")) for sentence in sentences]
        sentence_boundaries = [
            sum(sentence_boundaries[:i]) for i in range(1, len(sentence_boundaries))]

        # selecting human review from 1/10th to the 5/10th of the review
        selected_human_review_word = random.randint(int(num_of_words / 10), int(num_of_words / 2))
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

        kwargs = {"num_of_words_to_generate": num_of_words - selected_boundary,}

        partial_prompt = " ".join(words[:selected_boundary])
        updated_prompt = self.interpolate_prompt(data_item, partial_prompt, **kwargs)

        data_item["human_text"] = human_review
        data_item["cut_at_sentence"] = cut_at_sentence
        data_item["human_end_boundary"] = selected_boundary
        data_item["prompt"] = updated_prompt
        data_item["truncated_human_text"] = partial_prompt

        return data_item
