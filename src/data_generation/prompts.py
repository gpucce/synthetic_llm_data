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

XSUM_CHAT_CONTINUATION = """Act as an experienced journalist.

Continue the following Partial News Article complete it writing at least 300 words with a clear opinion. The written essay should look like human-written (please start writing the essay without any additional text).

Partial News Article:
{document}"""

XSUM_NON_CHAT_CONTINUATION = """{document}"""

ABSTRACTION_CHAT_CONTINUATION = """Answer as an experienced linguist.

Given two sentences, (SENT1) and (SENT2), both containing a word (WORD), please tell me if this word (WORD) is more abstract in the first sentence (SENT1), or in the second sentence (SENT2).
(Please answer by saying "sentence 1" or "sentence 2" only.)

{few_shots}SENT1: {text1}

SENT2: {text2}

WORD: {target_token}

ANSWER:"""

CHANGE_IT_CAMOSCIO_CHAT_CONTINUATION="""
Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n

### Istruzione:
Dato un titolo (TITOLO), scrivi un articolo di giornale di almeno 1000 parole il cui quello (TITOLO) sia il titolo.
Scrivi nel modo più elegante possibile.

### Input:
TITOLO = {title}

### Risposta: {article}
"""

CHANGE_IT_MISTRAL_NON_CHAT_CONTINUATION="""{title}
{article}"""

CHANGE_IT_MISTRAL_CHAT_CONTINUATION="""[INST]
Dato un titolo (TITOLO), scrivi un articolo di giornale di almeno 1000 parole di cui quello (TITOLO) sia il titolo.
Scrivi nel modo più elegante possibile e non ripetere il titolo (TITOLO). [/INST]

TITOLO: {title}

ARTICOLO: {article}
"""

CHANGE_IT_LLAMA_NON_CHAT_CONTINUATION="""{title}
{article}"""

CHANGE_IT_LLAMA_CHAT_CONTINUATION="""[INST] <<SYS>>
Scrivi come un esperto giornalista italiano che non si ripete mai.
<</SYS>>

Dato un titolo (TITOLO), scrivi un articolo di giornale di almeno 1000 parole di cui quello (TITOLO) sia il titolo.
Scrivi nel modo più elegante possibile e non ripetere il titolo (TITOLO). [/INST]

TITOLO: {title}

ARTICOLO: {article}
"""

CHANGE_IT_FINETUNE_CHAT_CONTINUATION=(
"""Dato il seguente titolo di un articolo di giornale scrivi l\'articolo.

### Titolo:
{title}

### Articolo:
{article}""")

INVALSI_LLAMA_CHAT_CONTINUATION = {
    'completa frase': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>> Dato il testo seguente:\nTesto:\n{testo}\n\n ragiona passo passo e completalo come richiesto nella domanda seguente indicando la risposta (risp) con questo formato <<RISPOSTA:{{risp}}>>[/INST]\n\nDomanda:\n\n{domanda}\n\nRisposta:\n",
    'multipla': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>> Dato il testo seguente:\nTesto:\n{testo}\n\n ragiona passo passo e scegli la risposta corretta alla domanda seguente indicando la risposta (risp) con questo formato <<RISPOSTA:{{risp}}>>:[/INST]\n\nDomanda:\n\n{domanda}\n\nRisposta:\n",
    'numero': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>> Dato il testo seguente:\nTesto:\n{testo}\n\n ragiona passo passo e rispondi con un numero come richiesto nella domanda seguente indicando la risposta (risp) con questo formato <<RISPOSTA:{{risp}}>>:[/INST]\n\nDomanda:\n\n{domanda}\n\nRisposta:\n",
    'vero/falso': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>> Dato il testo seguente:\nTesto:\n{testo}\n\n ragiona passo passo e indica se la frase seguente è vera o falsa indicando la risposta (risp) con questo formato <<RISPOSTA:{{risp}}>>:[/INST]\n\nFrase:\n\n{domanda}\n\nRisposta:\n",
}

PROMPT_REGISTRY = {
    "semeval_task_3" : {
        "peerread" : {
            **{model_name:PEERREAD_NON_CHAT_CONTINUATION for model_name
               in ["llama-2-7b-hf", "llama-2-13b-hf", "llama-2-70b-hf", "gpt2-hf"]},
            **{model_name:PEERREAD_CHAT_CONTINUATION for model_name
               in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf",
                   "gpt2-hf", "tiny-random-llama"]}
        },
        "outfox" : {
            model_name:OUTFOX_CHAT_CONTINUATION for model_name
                in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf",
                    "gpt2-hf", "tiny-random-llama"]
        },
        "xsum": {
            **{model_name:XSUM_CHAT_CONTINUATION for model_name
                in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]},
            **{model_name:XSUM_NON_CHAT_CONTINUATION for model_name
               in ["llama-2-7b-hf", "llama-2-13b-hf", "llama-2-70b-hf",
                   "gpt2-xl-hf", "gpt2-hf", "gpt2-small-hf", "gpt2-medium",
                   "tiny-random-llama"]}
        }
    },
    "wemb":{
        "abstraction": {
            model_name: ABSTRACTION_CHAT_CONTINUATION for model_name
                in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf",
                    "gpt2-hf", "tiny-random-llama", "gpt2-medium"]
        }
    },
    "ita_news":{
        "change_it": {
            **{model_name: CHANGE_IT_CAMOSCIO_CHAT_CONTINUATION for model_name
                in ["camoscio2_13b_v2", "camoscio2-70b-lora-hf_v2"]},
            **{model_name: CHANGE_IT_LLAMA_CHAT_CONTINUATION for model_name
                in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]},
            **{model_name: CHANGE_IT_MISTRAL_CHAT_CONTINUATION for model_name
                in ["Mixtral-8x7B-Instruct-v0.1", "Mistral-7B-Instruct-v0.2"]},
            **{model_name: CHANGE_IT_MISTRAL_NON_CHAT_CONTINUATION for model_name
                in ["Mistral-7B-v0.1", "Mixtral-8x7B-v0.1"]},
            **{model_name: CHANGE_IT_FINETUNE_CHAT_CONTINUATION for model_name
                in ["llama-13b_change_it", "llama-13b_change_it_3981_samples", 
                    "llama-13b_change_it_7962_samples", "llama-7b_change_it", 
                    "llama-7b_change_it_3981_samples", "llama-7b_change_it_7962_samples",
                    "mistral_change_it"]},
            **{model_name: CHANGE_IT_LLAMA_NON_CHAT_CONTINUATION for model_name
                in ["llama-2-7b-hf", "llama-2-13b-hf", "llama-2-70b-hf"]},
        }
    },
    "invalsi": {
        "invalsi_mate": {
            **{model_name: INVALSI_LLAMA_CHAT_CONTINUATION for model_name
                in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]}
        }
    }
}

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

    def interpolate_abstraction_prompt(self, data_item, partial_prompt, **kwargs):
        few_shots = ""
        if self.n_few_shots > 0:
            _few_shots = []
            few_shot_template = (
                "SENT1: {text1}\n\nSENT2: {text2}\n\nWORD: {target_token}\n\nANSWER:")
            assert few_shot_template in self.prompt, "Few shot template not found in the prompt"
            few_shot_template += "{answer}"
            for _ in range(self.n_few_shots):
                few_shot_data_item = self.data[random.randint(0, len(self.data) - 1)]
                while (few_shot_data_item["target_lemma1"] == data_item["target_lemma1"] or
                       few_shot_data_item["target_lemma2"] == data_item["target_lemma2"]):
                    few_shot_data_item = self.data[random.randint(0, len(self.data))]

                answer = "sentence 1" if few_shot_data_item["first_more_abstract"] else "sentence 2"
                answer += "\n\n"
                few_shot_prompt = few_shot_template.format(
                    text1=few_shot_data_item["text1"],
                    text2=few_shot_data_item["text2"],
                    target_token=few_shot_data_item["target_token"],
                    answer=answer
                )
                _few_shots.append(few_shot_prompt)
            few_shots = (f"Here are {self.n_few_shots} examples of the task:\n\n" +
                "".join(_few_shots))

        return self.prompt.format(
            text1=data_item["text1"],
            text2=data_item["text2"],
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
        elif self.preprocessing == "abstraction":
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
