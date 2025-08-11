def build_prompt_pairwise(
    source_sentence,
    target_sentence,
    full_doc1=None,
    full_doc2=None,
    prompt_mode=1,
    description=None,
    examples=None,
):
    prompt = ""
    prompt += f"Full Document 1:\n{full_doc1}\n\nFull Document 2:\n{full_doc2}\n\n"
    prompt += f"Source Sentence from Document 1:\n{source_sentence}\n\n"
    prompt += f"Target Sentence from Document 2:\n{target_sentence}\n\n"

    if prompt_mode == 1:
        prompt += "Determine if the target sentence is related to the source sentence. Answer with only 'True' or 'False'."
    elif prompt_mode == 2 and examples:
        prompt += "Here are some examples of related sentence pairs:\n"
        for source_ex, target_ex in examples:
            prompt += f" Source Sentence: {source_ex}\n   Target Sentence: {target_ex}\n"
        prompt += "\nDetermine if the target sentence is related to the source sentence using the provided related sentence pairs examples."
    elif prompt_mode == 3 and description:
        prompt += f"Relation description: {description}\n\nDetermine if the target sentence is related to the source sentence using the specified relation description."
    elif prompt_mode == 4 and description and examples:
        prompt += f"Relation description: {description}\n\nHere are some examples of related sentence pairs:\n"
        for source_ex, target_ex in examples:
            prompt += f" Source Sentence: {source_ex}\n   Target Sentence: {target_ex}\n"
        prompt += "\nDetermine if the target sentence is related to the source sentence using the specified relation description and the provided related sentence pairs examples."

    prompt += "\n\nReturn only a valid JSON object with one key 'related' whose value is either True or False."
    return prompt


def build_prompt_listwise(
    source_sentence,
    target_sentences,
    full_doc1=None,
    full_doc2=None,
    prompt_mode=1,
    description=None,
    positive_examples=None,
):
    prompt = ""
    prompt += f"Document 1:\n{full_doc1}\n\nDocument 2:\n{full_doc2}\n\n"
    prompt += f"Source Sentence from Document 1:\n{source_sentence}\n\n"
    prompt += f"Ranked Target Sentences from Document 2 (Sentence_ID: Sentence_text):\n{target_sentences}\n\n"

    if prompt_mode == 1:
        prompt += "Determine for each target sentence if it is related to the source sentence. Answer with only 'True' or 'False' for each sentence pair."
    elif prompt_mode == 2 and positive_examples:
        prompt += "Here are some examples of related sentence pairs:\n"
        for source_ex, target_ex in positive_examples:
            prompt += f" Source Sentence: {source_ex}\n   Target Sentence: {target_ex}\n"
        prompt += "\nDetermine for each target sentence if it is related to the source sentence using the provided related sentence pairs examples."
    elif prompt_mode == 3 and description:
        prompt += f"Relation Description: {description}\n\nDetermine for each target sentence if it is related to the source sentence using the specified relation description."
    elif prompt_mode == 4 and description and positive_examples:
        prompt += f"Relation Description: {description}\n\nHere are some examples of related sentence pairs:\n"
        for source_ex, target_ex in positive_examples:
            prompt += f" Source Sentence: {source_ex}\n   Target Sentence: {target_ex}\n"
        prompt += "\nDetermine for each target sentence if it is related to the source sentence using the specified relation description and the provided related sentence pairs examples."

    prompt += "\n\nReturn only a valid JSON object with the keys corresponding to the target sentence IDs and the values as True or False. For example: {'11': True, '4': False}"
    return prompt


def build_prompt_classification(
    source_sentence,
    full_doc1=None,
    full_doc2=None,
    prompt_mode=1,
    description=None,
    positive_examples=None,
):
    prompt = ""
    prompt += f"Document 1:\n{full_doc1}\n\nDocument 2:\n{full_doc2}\n\n"
    prompt += f"Source Sentence from Document 1:\n{source_sentence}\n\n"

    if prompt_mode == 1:
        prompt += "Determine for each target sentence in Document 2 if it is related to the source sentence. Answer with only 'True' or 'False' for each sentence pair."
    elif prompt_mode == 2 and positive_examples:
        prompt += "Here are some examples of related sentence pairs:\n"
        for source_ex, target_ex in positive_examples:
            prompt += f" Source Sentence: {source_ex}\n   Target Sentence: {target_ex}\n"
        prompt += "\nDetermine for each target sentence in Document 2 if it is related to the source sentence."
    elif prompt_mode == 3 and description:
        prompt += f"Relation Description: {description}\n\nDetermine for each target sentence in Document 2 if it is related to the source sentence."
    elif prompt_mode == 4 and description and positive_examples:
        prompt += f"Relation Description: {description}\n\nHere are some examples of related sentence pairs:\n"
        for source_ex, target_ex in positive_examples:
            prompt += f" Source Sentence: {source_ex}\n   Target Sentence: {target_ex}\n"
        prompt += "\nDetermine for each target sentence in Document 2 if it is related to the source sentence using the specified relation description and the provided related sentence pairs examples."

    prompt += "\n\nReturn only a valid JSON object with the keys corresponding to the target sentence IDs and the values as True or False. For example: {'11': True, '4': False}. Ensure you classify every target sentence"
    return prompt
