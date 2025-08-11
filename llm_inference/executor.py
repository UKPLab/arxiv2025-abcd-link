from tqdm import tqdm
from typing import Dict, Any

from llm_inference.chat_utils import (
    SentenceRelations,
    build_chat_prompt,
    get_guided_sampling_params,
)


def classify_dataset_batched_global(
    dataset_prompts: Dict[str, Dict[str, Any]],
    model_name: str,
    llm,
    tokenizer,
    prompt_setup: str,
    batch_size: int = 32,
):
    """
    Run vLLM classification for a whole dataset in batches.
    """
    # Flatten all prompts into (idx, source_id, prompt_text)
    all_prompts = []
    for idx, prompts in dataset_prompts.items():
        for source_key, prompt_data in prompts.items():
            raw_prompt = prompt_data["prompt"]
            chat_prompt = build_chat_prompt(raw_prompt, tokenizer, prompt_setup)
            all_prompts.append((idx, source_key, chat_prompt))

    # Prepare result structure
    llm_classified = {idx: {"links": {}} for idx in dataset_prompts}

    # Helper: batching
    def chunk(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    # Sampling params
    sampling_params = get_guided_sampling_params(SentenceRelations)

    # Run LLM in batches
    for batch in tqdm(list(chunk(all_prompts, batch_size)), desc="Classifying prompts"):
        texts = [entry[2] for entry in batch]
        outputs = llm.generate(texts, sampling_params, use_tqdm=False)

        for (idx, source_key, _), output in zip(batch, outputs):
            output_text = output.outputs[0].text.strip()
            try:
                parsed = SentenceRelations.model_validate_json(output_text)
                rel_dict = {r.sentence_id: r.related for r in parsed.relations}
            except Exception as e:
                print(f"Failed to parse response for {idx} -> {source_key}: {e}")
                print("Raw output:\n", output_text)
                rel_dict = {}

            llm_classified[idx]["links"][source_key] = rel_dict

    return llm_classified
