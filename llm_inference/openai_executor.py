import json
from tqdm import tqdm
from llm_inference.openai_utils import SentenceRelations, get_openai_system_message


def classify_with_openai(prompt: str, model_name: str, client, system_message: str):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    response = client.beta.chat.completions.parse(
        model=model_name,
        messages=messages,
        temperature=0.4,
        response_format=SentenceRelations,
    )

    return response


def extract_openai_relations(response):
    relations = response.choices[0].message.parsed.model_dump()["relations"]
    return {r["sentID"]: r["related"] for r in relations}


def classify_dataset_openai(dataset_prompts, model_name, client, prompt_setup: str):
    system_message = get_openai_system_message(prompt_setup)
    llm_classified_datasets = {}

    for idx, prompts in tqdm(dataset_prompts.items(), desc=f"OpenAI: {model_name}"):
        llm_classified_datasets[idx] = {"links": {}}
        for source, prompt_data in prompts.items():
            prompt = prompt_data["prompt"]
            try:
                response = classify_with_openai(prompt, model_name, client, system_message)
                rel_dict = extract_openai_relations(response)
            except Exception as e:
                print(f"Error processing {idx} -> {source}: {e}")
                rel_dict = {}
            llm_classified_datasets[idx]["links"][source] = rel_dict

    return llm_classified_datasets
