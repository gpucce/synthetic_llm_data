# pylint: disable=line-too-long

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
