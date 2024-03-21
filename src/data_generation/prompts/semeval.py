# pylint: disable=line-too-long

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
2. You only complete the partial review and not write it from the beginning. [/INST]

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
