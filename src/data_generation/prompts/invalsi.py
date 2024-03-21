# pylint: disable=line-too-long

INVALSI_LLAMA_CHAT = {
    'completa frase': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>>\n\nDato un testo (TESTO) completalo come richiesto e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'multipla': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>>\n\nDato un testo (TESTO) scegli la risposta alla domanda (DOMANDA) seguente tra quelle disponibili e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'numero': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>>\n\nDato un testo (TESTO) calcola il numero come richiesto nella domanda (DOMANDA) all fine dai come risposta finale solo un numero. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'vero/falso': "[INST] <<SYS>>\nWrite in Italian as a very deductive and precise student.\n<</SYS>>\n\nDato un testo (TESTO) indica se la frase (FRASE) che lo segue è vera o falsa e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nFRASE:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
}

INVALSI_LLAMANTINO_CHAT = {
    'completa frase': "[INST] <<SYS>>\nScrivi come uno studente molto accurato e preciso.\n<</SYS>>\n\nDato un testo (TESTO) completalo come richiesto e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'multipla': "[INST] <<SYS>>\nScrivi come uno studente molto accurato e preciso.\n<</SYS>>\n\nDato un testo (TESTO) scegli la risposta alla domanda (DOMANDA) seguente tra quelle disponibili e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'numero': "[INST] <<SYS>>\nScrivi come uno studente molto accurato e preciso.\n<</SYS>>\n\nDato un testo (TESTO) calcola il numero come richiesto nella domanda (DOMANDA) all fine dai come risposta finale solo un numero. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'vero/falso': "[INST] <<SYS>>\nScrivi come uno studente molto accurato e preciso.\n<</SYS>>\n\nDato un testo (TESTO) indica se la frase (FRASE) che lo segue è vera o falsa e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nFRASE:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
}

INVALSI_MISTRAL_INSTRUCT = {
    'completa frase': "[INST] Dato un testo (TESTO) completalo come richiesto e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'multipla': "[INST] Dato un testo (TESTO) scegli la risposta alla domanda (DOMANDA) seguente tra quelle disponibili e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'numero': "[INST] Dato un testo (TESTO) calcola il numero come richiesto nella domanda (DOMANDA) all fine dai come risposta finale solo il numero. [/INST]\n\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
    'vero/falso': "[INST] Dato un testo (TESTO) indica se la frase (FRASE) che lo segue è vera o falsa e dai una risposta finale. [/INST]\n\nTESTO:\n\n{testo}\n\nFRASE:\n\n{domanda}\n\nRagioniamo passo passo", # pylint: disable=line-too-long
}

INVALSI_CAMOSCIO = {
    'completa frase': "Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n### Istruzione:\nDato un testo (TESTO) completalo come richiesto e dai una risposta finale.\n\n{few_shots}### Input:\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\n### Risposta:\nRagioniamo passo passo", # pylint: disable=line-too-long
    'multipla': "Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n### Istruzione:\n Dato un testo (TESTO) scegli la risposta alla domanda (DOMANDA) seguente tra quelle disponibili e dai una risposta finale.\n\n{few_shots}### Input:\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\n### Risposta:\nRagioniamo passo passo", # pylint: disable=line-too-long
    'numero': "Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n### Istruzione:\nDato un testo (TESTO) calcola il numero come richiesto nella domanda (DOMANDA) alla fine dai come risposta finale solo il numero.\n\n{few_shots}### Input:\nTESTO:\n\n{testo}\n\nDOMANDA:\n\n{domanda}\n\n### Risposta:\nRagioniamo passo passo", # pylint: disable=line-too-long
    'vero/falso': "Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n### Istruzione:\nDato un testo (TESTO) indica se la frase (FRASE) che lo segue è vera o falsa e dai una risposta finale.\n\n{few_shots}### Input:\nTESTO:\n\n{testo}\n\nFRASE:\n\n{domanda}\n\n### Risposta:\nRagioniamo passo passo", # pylint: disable=line-too-long
}
