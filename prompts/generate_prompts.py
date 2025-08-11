import os
import json
from prompts.builder import (
    build_prompt_pairwise,
    build_prompt_listwise,
    build_prompt_classification,
)

# Constants
DATASETS = ["news_ecb", "news_synth", "reviews_synth", "reviews_f1000"]
RELATION_DESCRIPTIONS = {
    "news_ecb": "Two sentences that convey the same factual event, differing in expression, vocabulary, or degree of detail.",
    "news_synth": "Two sentences that convey the same factual event, differing in expression, vocabulary, or degree of detail.",
    "reviews_synth": "The source sentence is a reviewer’s comment that clarifies, expands on, or critiques the target sentence which is from a research paper.",
    "reviews_f1000": "The source sentence is a reviewer’s comment that clarifies, expands on, or critiques the target sentence which is from a research paper.",
}
PROMPT_MODES = {
    1: "no_instruction",
    2: "examples",
    3: "description",
    4: "description_and_examples",
}
BEST_MODEL = "dragon_plus"
TARGET_LIMITS = {"news": 10, "reviews": 20}

# Load positive examples
with open("./data/positive_examples.json") as f:
    positive_examples = json.load(f)

# Create output directory
os.makedirs("./prompts", exist_ok=True)
os.makedirs("./data", exist_ok=True)
os.makedirs("./data/prompts_json", exist_ok=True)

# Load datasets, corpora, and retrieval predictions
datasets = {}
corpora = {}
predictions = {}

for dataset in DATASETS:
    with open(f"./datasets/{dataset}/{dataset}_links.json") as f:
        datasets[dataset] = json.load(f)
    with open(f"./datasets/{dataset}/docs.json") as f:
        corpora[dataset] = json.load(f)
    with open(f"./predictions/{dataset}_{BEST_MODEL}_scores.json") as f:
        predictions[dataset] = json.load(f)

# Main prompt generation loop
for prompt_mode in PROMPT_MODES.keys():
    for prompt_setup in ["pairwise", "listwise", "classification"]:
        for dataset_name in DATASETS:
            dataset = datasets[dataset_name]
            corpus = corpora[dataset_name]
            preds = predictions[dataset_name]
            out_data = {}

            limit = TARGET_LIMITS["reviews"] if "reviews" in dataset_name else TARGET_LIMITS["news"]
            relation = RELATION_DESCRIPTIONS[dataset_name]
            examples = positive_examples.get(dataset_name, [])

            for idx, data in dataset.items():
                doc1 = corpus[data["doc1"]]
                doc2 = corpus[data["doc2"]]
                entry_prompts = {}

                for source in data["links"]:
                    source_sentence = doc1[source]
                    ranked_ids = list(preds[idx]["links"][source].keys())[:limit]
                    target_sents = {tid: doc2[tid] for tid in ranked_ids}

                    if prompt_setup == "pairwise":
                        for target_id, target_sent in target_sents.items():
                            prompt = build_prompt_pairwise(
                                source_sentence,
                                target_sent,
                                full_doc1=doc1,
                                full_doc2=doc2,
                                prompt_mode=prompt_mode,
                                description=relation,
                                examples=examples,
                            )
                            entry_prompts[target_id] = {"prompt": prompt}

                    elif prompt_setup == "listwise":
                        prompt = build_prompt_listwise(
                            source_sentence,
                            target_sents,
                            full_doc1=doc1,
                            full_doc2=doc2,
                            prompt_mode=prompt_mode,
                            description=relation,
                            positive_examples=examples,
                        )
                        entry_prompts[source] = {"prompt": prompt}

                    elif prompt_setup == "classification":
                        prompt = build_prompt_classification(
                            source_sentence,
                            full_doc1=doc1,
                            full_doc2=doc2,
                            prompt_mode=prompt_mode,
                            description=relation,
                            positive_examples=examples,
                        )
                        entry_prompts[source] = {"prompt": prompt}

                out_data[idx] = entry_prompts

            # Save to file
            file_path = f"./data/prompts_json/{dataset_name}_prompts_{prompt_setup}_{PROMPT_MODES[prompt_mode]}.json"
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(out_data, f, indent=4, ensure_ascii=False)
            print(f"Saved: {file_path}")