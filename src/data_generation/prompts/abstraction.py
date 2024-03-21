# pylint: disable=line-too-long

_ABSTRACTION = """Given two sentences, (SENT1) and (SENT2), both containing a word (WORD), please tell me if this word (WORD) is more abstract in the first sentence (SENT1), or in the second sentence (SENT2).
(Please answer by saying "sentence 1" or "sentence 2" only.) [/INST]

{few_shots}SENT1: {text1}

SENT2: {text2}

WORD: {target_token}

ANSWER:"""


ABSTRACTION_LLAMA_CHAT = """[INST] <<SYS>>
Answer as an experienced linguist.
<</SYS>>

""" + _ABSTRACTION

ABSTRACTION_MISTRAL_CHAT = """[INST] """ + _ABSTRACTION

_ABSTRACTION_REGRESSION = """Given a sentences (SENT) and a word (WORD), please assign a rank (RANK) from 1 to 5 to the abstractness of the word (WORD) in the sentences (SENT).

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


ABSTRACTION_REGRESSION_LLAMA_CHAT = """[INST] <<SYS>>
Answer as an experienced linguist.
<</SYS>>

""" + _ABSTRACTION_REGRESSION

ABSTRACTION_REGRESSION_MISTRAL_CHAT = "[INST] " + _ABSTRACTION_REGRESSION

_INCLUSIVENESS= """Given two sentences, (SENT1) and (SENT2), both containing a word (WORD), please tell me if this word (WORD) is more inclusive in the first sentence (SENT1), or in the second sentence (SENT2).
Inclusiveness indicates the number of things the word (WORD) can refer to.

(Please answer by saying "sentence 1" or "sentence 2" only.) [/INST]

{few_shots}SENT1: {text1}

SENT2: {text2}

WORD: {target_token}

ANSWER:"""


INCLUSIVENESS_LLAMA_CHAT = """[INST] <<SYS>>
Answer as an experienced linguist.
<</SYS>>

""" + _INCLUSIVENESS

INCLUSIVENESS_MISTRAL_CHAT = """[INST] """ + _INCLUSIVENESS

_INCLUSIVENESS_REGRESSION= """Given a sentences (SENT) and a word (WORD), please assign a rank (RANK) from 1 to 5 to the inclusiveness of the word (WORD) in the sentences (SENT).
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


INCLUSIVENESS_REGRESSION_LLAMA_CHAT = """[INST] <<SYS>>
Answer as an experienced linguist.
<</SYS>>

""" + _INCLUSIVENESS_REGRESSION

INCLUSIVENESS_REGRESSION_MISTRAL_CHAT = """[INST] """ + _INCLUSIVENESS_REGRESSION
