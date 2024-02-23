
LLAMA_CHAT_MODELS = [
    "llama-2-7b-chat-hf", 
    "llama-2-13b-chat-hf", 
    "llama-2-70b-chat-hf", 
    "tiny-random-llama"
]

LLAMA_MODELS = [
    "llama-2-7b-hf", 
    "llama-2-13b-hf", 
    "llama-2-70b-hf",
    "tiny-random-llama"
]

PEERREAD_NON_CHAT="""Title of the paper:
{paper_title}

Abstract of the paper:
{paper_abstract}

Review:
{partial_review}"""

PEERREAD_LLAMA_CHAT="""[INST] <<SYS>>
Act as an experiend scientis that usually does peer-review of papers 
<</SYS>>

Task: Complete a partially-written peer review of the following paper.

Make sure that:
1. The completion is of at least {num_of_words} words.
2. You only complete the partial review and not write it from the beginning.

Title of the paper:
{paper_title}

Abstract of the paper:
{paper_abstract}

Review:
{partial_review}"""

OUTFOX_LLAMA_CHAT = """[INST] <<SYS>>
Act as an experienced essay writer. 
<</SYS>>

Given the following Problem statement (essay topic) complete the following Partial Essay writing at least 318 words with a clear opinion. The written essay should look like human-written (please start writing the essay without any additional text). [/INST]

Problem statement (essay topic):
{essay_topic}

Partial Essay:
{partial_essay}"""

XSUM_LLAMA_CHAT = """[INST] <<SYS>>
Act as an experienced journalist.
<</SYS>>

Continue the following Partial News Article complete it writing at least 300 words with a clear opinion. The written essay should look like human-written (please start writing the essay without any additional text). [/INST]

Partial News Article:
{document}"""

XSUM_NON_CHAT = """{document}"""

ABSTRACTION_LLAMA_CHAT = """[INST] <<SYS>>
Answer as an experienced linguist.
<</SYS>>

Given two sentences, (SENT1) and (SENT2), both containing a word (WORD), please tell me if this word (WORD) is more abstract in the first sentence (SENT1), or in the second sentence (SENT2).
(Please answer by saying "sentence 1" or "sentence 2" only.) [/INST]

{few_shots}SENT1: {text1}

SENT2: {text2}

WORD: {target_token}

ANSWER:"""

ABSTRACTION_REGRESSION_LLAMA_CHAT = """[INST] <<SYS>>
Answer as an experienced linguist.
<</SYS>>

Given a sentences (SENT) and a word (WORD), please assign a rank (RANK) from 1 to 5 to the abstractness of the word (WORD) in the sentences (SENT).

Ranking description:

1 - The word (WORD) is extremely concrete (a specific dog)
2 - The word (WORD) is slightly concrete (a dog of a given breed)
3 - The word (WORD) is neutral (one of many dogs)
4 - The word (WORD) is slightly abstract (a dog can be a pet)
5 - The word (WORD) is extremely abstract (a dog is a mammal)

Answer by only writing: "the rank is: (RANK)". [/INST]

{few_shots}SENT: {text1}

WORD: {target_token}

ANSWER:"""

INCLUSIVENESS_LLAMA_CHAT = """[INST] <<SYS>>
Answer as an experienced linguist.
<</SYS>>

Given two sentences, (SENT1) and (SENT2), both containing a word (WORD), please tell me if this word (WORD) is more inclusive in the first sentence (SENT1), or in the second sentence (SENT2).
Inclusiveness indicates the number of things the word (WORD) can refer to.

(Please answer by saying "sentence 1" or "sentence 2" only.) [/INST]

{few_shots}SENT1: {text1}

SENT2: {text2}

WORD: {target_token}

ANSWER:"""

INCLUSIVENESS_REGRESSION_LLAMA_CHAT = """[INST] <<SYS>>
Answer as an experienced linguist.
<</SYS>>

Given a sentences (SENT) and a word (WORD), please assign a rank (RANK) from 1 to 5 to the inclusiveness of the word (WORD) in the sentences (SENT).
Inclusiveness indicates the number of things the word (WORD) can refer to.


Ranking description:

1 - The word (WORD) is extremely specific (a specific dog)
2 - The word (WORD) is slightly specific (a dog of a given breed)
3 - The word (WORD) is neutral (one of many dogs)
4 - The word (WORD) is slightly inclusive (a dog can be a pet)
5 - The word (WORD) is extremely inclusive (a dog is a mammal)

Answer by only writing: "the rank is: (RANK)". [/INST]

{few_shots}SENT: {text1}

WORD: {target_token}

ANSWER:"""

CHANGE_IT_CAMOSCIO_CHAT = """
Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n

### Istruzione:
Dato un titolo (TITOLO), scrivi un articolo di giornale di almeno 1000 parole il cui quello (TITOLO) sia il titolo.
Scrivi nel modo più elegante possibile.

### Input:
TITOLO = {title}

### Risposta: {article}
"""

CHANGE_IT_MISTRAL_NON_CHAT="""{title}
{article}"""

CHANGE_IT_MISTRAL_CHAT="""[INST]
Dato un titolo (TITOLO), scrivi un articolo di giornale di almeno 1000 parole di cui quello (TITOLO) sia il titolo.
Scrivi nel modo più elegante possibile e non ripetere il titolo (TITOLO). [/INST]

TITOLO: {title}

ARTICOLO: {article}
"""

CHANGE_IT_LLAMA_NON_CHAT = """{title}
{article}"""

CHANGE_IT_LLAMA_CHAT="""[INST] <<SYS>>
Scrivi come un esperto giornalista italiano che non si ripete mai.
<</SYS>>

Dato un titolo (TITOLO), scrivi un articolo di giornale di almeno 1000 parole di cui quello (TITOLO) sia il titolo.
Scrivi nel modo più elegante possibile e non ripetere il titolo (TITOLO). [/INST]

TITOLO: {title}

ARTICOLO: {article}
"""

CHANGE_IT_FINETUNE_CHAT = (
"""Dato il seguente titolo di un articolo di giornale scrivi l\'articolo.

### Titolo:
{title}

### Articolo:
{article}""")

INVALSI_LLAMA_CHAT = {
    'completa frase': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>>\n\nDato il testo seguente:\nTesto:\n{testo}\n\n ragiona passo passo e completalo come richiesto nella domanda seguente indicando la risposta (risp) con questo formato <<RISPOSTA:{{risp}}>>[/INST]\n\nDomanda:\n\n{domanda}\n\nRisposta:\n", # pylint: disable=line-too-long
    'multipla': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>>\n\nDato il testo seguente:\nTesto:\n{testo}\n\n ragiona passo passo e scegli la risposta corretta alla domanda seguente indicando la risposta (risp) con questo formato <<RISPOSTA:{{risp}}>>:[/INST]\n\nDomanda:\n\n{domanda}\n\nRisposta:\n", # pylint: disable=line-too-long
    'numero': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>>\n\nDato il testo seguente:\nTesto:\n{testo}\n\n ragiona passo passo e rispondi con un numero come richiesto nella domanda seguente indicando la risposta (risp) con questo formato <<RISPOSTA:{{risp}}>>:[/INST]\n\nDomanda:\n\n{domanda}\n\nRisposta:\n", # pylint: disable=line-too-long
    'vero/falso': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>>\n\nDato il testo seguente:\nTesto:\n{testo}\n\n ragiona passo passo e indica se la frase seguente è vera o falsa indicando la risposta (risp) con questo formato <<RISPOSTA:{{risp}}>>:[/INST]\n\nFrase:\n\n{domanda}\n\nRisposta:\n", # pylint: disable=line-too-long
}

PROMPT_REGISTRY = {
    "semeval_task_3" : {
        "peerread" : {
            **{model_name:PEERREAD_NON_CHAT for model_name in LLAMA_MODELS + ["gpt2-hf"]},
            **{model_name:PEERREAD_LLAMA_CHAT for model_name in LLAMA_CHAT_MODELS}
        },
        "outfox" : {
            model_name:OUTFOX_LLAMA_CHAT for model_name in LLAMA_CHAT_MODELS
        },
        "xsum": {
            **{model_name:XSUM_LLAMA_CHAT for model_name in LLAMA_CHAT_MODELS},
            **{model_name:XSUM_NON_CHAT for model_name
               in LLAMA_MODELS +["gpt2-xl-hf", "gpt2-hf", "gpt2-small-hf", "gpt2-medium",
                   "llama-7b_xsum", "mistral_xsum"]}
        }
    },
    "wemb":{
        "abstraction": {
            model_name: ABSTRACTION_LLAMA_CHAT for model_name in LLAMA_CHAT_MODELS},
        "abstraction_regression": {
            model_name: ABSTRACTION_REGRESSION_LLAMA_CHAT for model_name in LLAMA_CHAT_MODELS},
        "inclusiveness": {
            model_name: INCLUSIVENESS_LLAMA_CHAT for model_name in LLAMA_CHAT_MODELS},
        "inclusiveness_regression": {
            model_name: INCLUSIVENESS_REGRESSION_LLAMA_CHAT for model_name in LLAMA_CHAT_MODELS},
    },
    "ita_news":{
        "change_it": {
            **{model_name: CHANGE_IT_CAMOSCIO_CHAT for model_name
                in ["camoscio2_13b_v2", "camoscio2-70b-lora-hf_v2"]},
            **{model_name: CHANGE_IT_LLAMA_CHAT for model_name in LLAMA_CHAT_MODELS},
            **{model_name: CHANGE_IT_MISTRAL_CHAT for model_name
                in ["Mixtral-8x7B-Instruct-v0.1", "Mistral-7B-Instruct-v0.2"]},
            **{model_name: CHANGE_IT_MISTRAL_NON_CHAT for model_name
                in ["Mistral-7B-v0.1", "Mixtral-8x7B-v0.1"]},
            **{model_name: CHANGE_IT_FINETUNE_CHAT for model_name
                in ["llama-13b_change_it", "llama-13b_change_it_3981_samples",
                    "llama-13b_change_it_7962_samples", "llama-7b_change_it", 
                    "llama-7b_change_it_3981_samples", "llama-7b_change_it_7962_samples",
                    "mistral_change_it"]},
            **{model_name: CHANGE_IT_LLAMA_NON_CHAT for model_name in LLAMA_MODELS}}
    },
    "invalsi": {
        "invalsi_mate": {
            **{model_name: INVALSI_LLAMA_CHAT for model_name
                in LLAMA_CHAT_MODELS}}
    }
}
