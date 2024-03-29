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


_ABSTRACTION_REGRESSION_ITA = """Data una frase (FRASE) e una parola (PAROLA), assegna un valore da 1 a 5 che indica il livello (LIVELLO) di astrazione della parola (PAROLA) nella frase (FRASE).

Descrizione dei livelli:

1 - La parola (PAROLA) è estremamente concreta (uno specifico cane)
2 - La parola (PAROLA) è abbastanza concreta (un cane di una razza specifica)
3 - La parola (PAROLA) è neutra (un cane tra tanti)
4 - La parola (PAROLA) è abbastanza astratta (un cane da compagnia)
5 - La parola (PAROLA) è estremamente astratta (un qualsiasi cane)

Rispondi dicendo solo: Il livello è: (LIVELLO). [/INST]

{few_shots}FRASE: {text1}

PAROLA: {target_token}

RISPOSTA:"""



ABSTRACTION_REGRESSION_LLAMA_CHAT_ITA = """[INST] <<SYS>>
Answer in Italian as an experienced linguist.
<</SYS>>

""" + _ABSTRACTION_REGRESSION_ITA

ABSTRACTION_REGRESSION_MISTRAL_CHAT_ITA = "[INST] " + _ABSTRACTION_REGRESSION_ITA

_INCLUSIVENESS_REGRESSION_ITA = """Data una frase (FRASE) e una parola (PAROLA), assegna un valore da 1 a 5 che indica il livello (LIVELLO) di inclusività della parola (PAROLA) nella frase (FRASE).

Descrizione dei livelli:

1 - La parola (PAROLA) è estremamente specifica (uno specifico cane)
2 - La parola (PAROLA) è abbastanza specifica (un cane di una razza specifica)
3 - La parola (PAROLA) è neutra (un cane tra tanti)
4 - La parola (PAROLA) è abbastanza inclusiva (un cane da compagnia)
5 - La parola (PAROLA) è estremamente inclusiva (un qualsiasi cane)

Rispondi dicendo solo: Il livello è: (LIVELLO). [/INST]

{few_shots}FRASE: {text1}

PAROLA: {target_token}

RISPOSTA:"""


INCLUSIVENESS_REGRESSION_LLAMA_CHAT_ITA = """[INST] <<SYS>>
Answer in Italian as an experienced linguist.
<</SYS>>

""" + _INCLUSIVENESS_REGRESSION_ITA

INCLUSIVENESS_REGRESSION_MISTRAL_CHAT_ITA = """[INST] """ + _INCLUSIVENESS_REGRESSION_ITA