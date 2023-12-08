def camoscio_preprocessing_function(inp):
    """Format the text string."""
    ITA_PROMPT_FORMAT_NO_INPUT = "Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n### Istruzione:\n{instruction}\n\n### Risposta:\n"
    ITA_PROMPT_FORMAT = "Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n### Istruzione:\n{instruction}\n\n### Input:\n{input}\n\n### Risposta:\n"
    try:
        if inp["input"] != "":
            prompt = ITA_PROMPT_FORMAT.format(
                instruction=inp["instruction"], input=inp["input"]
            )
        else:
            prompt = ITA_PROMPT_FORMAT_NO_INPUT.format(
                instruction=inp["instruction"])
        response = inp["output"]
    except Exception as e:
        raise ValueError(
            f"Unable to extract prompt/response from {inp=}") from e
    #     return {'prompt': prompt, 'response': response}
    prompt = prompt + response
    return {"text": prompt}
