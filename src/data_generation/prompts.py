# pylint: disable=unused-argument
import random
from pathlib import Path

from .utils import split_by_full_stop
from .prompt_list import PROMPT_REGISTRY


class PromptPreprocessor():
    def __init__(self, args, data):
        self.preprocessing = args.preprocessing
        self.model_name = Path(args.name_or_path).name
        self.human_key = args.human_key
        self.project = args.project
        self.prompt = PROMPT_REGISTRY[self.project][self.preprocessing][self.model_name]
        self.split_at_random_length = args.split_at_random_length
        self.selected_boundary = args.selected_boundary
        self.n_few_shots = args.n_few_shots
        self.data = data

    def interpolate_peerread_prompt(
        self, data_item, partial_prompt, num_of_words_to_generate, **kwargs):

        return self.prompt.format(
            paper_title=data_item["title"],
            paper_abstract=data_item["abstract"],
            partial_review=partial_prompt,
            num_of_words=num_of_words_to_generate,
        )

    def interpolate_xsum_prompt(self, data_item, partial_prompt, **kwargs):
        return self.prompt.format(document=partial_prompt)

    def interpolate_abstraction_prompt(self, data_item, **kwargs):
        few_shots = ""
        if self.n_few_shots > 0:
            _few_shots = []
            _past_idxs = set()

            if self.preprocessing in ["abstraction", "inclusiveness"]:
                few_shot_template = (
                    "SENT1: {text1}\n\nSENT2: {text2}\n\nWORD: {target_token}\n\nANSWER:")
                assert few_shot_template in self.prompt, "Few shot template not found in the prompt"
                few_shot_template += "{answer}"
            elif self.preprocessing in ["abstraction_regression", "inclusiveness_regression"]:
                few_shot_template = (
                    "SENT: {text1}\n\nWORD: {target_token}\n\nANSWER:")
                assert few_shot_template in self.prompt, "Few shot template not found in the prompt"
                few_shot_template += "\"the rank is: {answer}\""


            for _ in range(self.n_few_shots):
                idx = random.randint(0, len(self.data) - 1)
                while idx in _past_idxs:
                    idx = random.randint(0, len(self.data) - 1)
                _past_idxs.add(idx)
                few_shot_data_item = self.data[idx]

                if self.preprocessing in ["abstraction", "inclusiveness"]:
                    _key = ("first_more_abstract"
                            if "abstraction" in self.preprocessing else "first_more_inclusive")
                    answer = "sentence 1" if few_shot_data_item[_key] else "sentence 2"
                    few_shot_prompt = few_shot_template.format(
                        text1=few_shot_data_item["text1"],
                        text2=few_shot_data_item["text2"],
                        target_token=few_shot_data_item["target_token"],
                        answer=answer
                    )
                elif self.preprocessing in ["abstraction_regression", "inclusiveness_regression"]:
                    # TODO: this is hardcoded might need changing
                    _key = "abs_mean" if "abstraction" in self.preprocessing else "inc_mean"
                    answer = str(round(5 * few_shot_data_item[_key])).replace("0", "1")
                    few_shot_prompt = few_shot_template.format(
                        text1=few_shot_data_item["text"],
                        target_token=few_shot_data_item["target_token"],
                        answer=answer
                    )
                few_shot_prompt += "\n\n"
                _few_shots.append(few_shot_prompt)

            few_shots = (f"Here are {self.n_few_shots} examples of the task:\n\n" +
                "".join(_few_shots))

        if self.preprocessing in ["abstraction", "inclusiveness"]:
            return self.prompt.format(
                text1=data_item["text1"],
                text2=data_item["text2"],
                target_token=data_item["target_token"],
                few_shots = few_shots
            )
        elif self.preprocessing in ["abstraction_regression", "inclusiveness_regression"]:
            return self.prompt.format(
                text1=data_item["text"],
                target_token=data_item["target_token"],
                few_shots = few_shots
            )


    def interpolate_outfox_prompt(self, data_item, partial_prompt, **kwargs):
        return self.prompt.format(
            essay_topic=data_item["topic"],
            partial_essay=partial_prompt
        )

    def interpolate_changeit(self, data_item, partial_prompt, **kwargs):
        return self.prompt.format(
            title=data_item["headline"],
            article=partial_prompt)

    def interpolate_invalsi(self, data_item, partial_prompt, **kwargs):
        return self.prompt[data_item["tipo"]].format(
            testo=data_item["testo"], domanda=partial_prompt)

    def interpolate_prompt(self, data_item, partial_prompt, **kwargs):
        if self.preprocessing == "peerread":
            return self.interpolate_peerread_prompt(
                data_item, partial_prompt, **kwargs)
        elif self.preprocessing == "outfox":
            return self.interpolate_outfox_prompt(
                data_item, partial_prompt, **kwargs)
        elif self.preprocessing == "xsum":
            return self.prompt.format(data_item, document=partial_prompt, **kwargs)
        elif self.preprocessing in [
            "abstraction", "inclusiveness", "abstraction_regression", "inclusiveness_regression"]:
            return self.interpolate_abstraction_prompt(data_item, **kwargs)
        elif self.preprocessing == "change_it":
            return self.interpolate_changeit(data_item, partial_prompt=partial_prompt)
        elif self.preprocessing == "invalsi_mate":
            return self.interpolate_invalsi(data_item, partial_prompt=partial_prompt)

        raise ValueError(f"Unknown formatting {self.preprocessing}")

    def get_boundary(self, num_of_words, cut_at_sentence, words, sentences):
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

        return selected_boundary

    def __call__(self, data_item, **kwargs):
        """
        A part of human review could be cut off 
        1. randomly at a word present in 1/10th to 5/10th of the review.
        2. randomly at a closest sentence boundary in the 1/10th to 5/10th of the review.
        """

        original_human_review = data_item[self.human_key]
        human_review = self.interpolate_prompt(data_item, partial_prompt=original_human_review)

        cut_at_sentence = random.randint(0, 2) > 1

        sentences = split_by_full_stop(human_review)
        words = human_review.split(" ")

        num_of_words = len(words)

        selected_boundary = self.selected_boundary
        if self.split_at_random_length:
            selected_boundary = self.get_boundary(
                num_of_words, cut_at_sentence, words, sentences)
        selected_boundary = min(selected_boundary, num_of_words)

        partial_prompt = " ".join(words[:selected_boundary])

        data_item["human_text"] = human_review
        data_item["cut_at_sentence"] = cut_at_sentence
        data_item["human_end_boundary"] = selected_boundary
        data_item["prompt"] = partial_prompt
        data_item["truncated_human_text"] = original_human_review

        return data_item
