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
               in ["llama-2-7b-hf", "llama-2-13b-hf", "llama-2-70b-hf"]},
            **{model_name:PEERREAD_CHAT_CONTINUATION for model_name
               in ["llama-2-7b-chat-hf", "llama-2-13b-chat-hf", "llama-2-70b-chat-hf"]}
            },
        "outfox" : {
            "gpt4": OUTFOX_CHAT_CONTINUATION,
        }
    }
}
