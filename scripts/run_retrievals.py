import os
import json
import time
import spacy
import torch
from tqdm import tqdm

from retrieval.models import load_model
from retrieval.scorers import (
    calculate_bm25_scores,
    calculate_splade_scores,
    calculate_dense_scores,
    calculate_bgem3_dense_scores,
    calculate_bgem3_sparse_scores,
    calculate_contriever_similarity
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load datasets
DATASETS = ["news_ecb", "news_synth", "reviews_synth", "reviews_f1000"]
datasets = {}
datasets_corpus = {}

for dataset in DATASETS:
    with open(f"./datasets/{dataset}/{dataset}_links.json") as f:
        datasets[dataset] = json.load(f)
    with open(f"./datasets/{dataset}/docs.json") as f:
        datasets_corpus[dataset] = json.load(f)
    print(f"Loaded {dataset} with {len(datasets[dataset])} examples.")

# Load spaCy tokenizer once
nlp = spacy.load("en_core_web_md")

# Define model types
models = {
    "BM25": "sparse",
    "splade": "sparse",
    "bgem3-sparse": "sparse",
    "all-mpnet": "dense",
    "SFR": "dense",
    "bgem3-dense": "dense",
    "contriever": "dense",
    "dragon_plus": "dense",
    "ms_marco_MiniLM": "cross-encoder",
    "bge-reranker": "cross-encoder",
}

# Ensure predictions folder exists
os.makedirs("./predictions", exist_ok=True)

# Main execution
if __name__ == "__main__":
    for dataset_name in DATASETS:
        doc_pairs = datasets[dataset_name]
        corpus = datasets_corpus[dataset_name]

        print("=" * 50)
        print(f"Processing dataset: {dataset_name}")
        start_time = time.time()

        for model_name, model_type in models.items():
            print(f"\n>>> Model: {model_name} ({model_type})")
            output_path = f"./predictions/{dataset_name}_{model_name}_scores.json"

            # Skip if already exists
            if os.path.exists(output_path):
                print(f"Already exists: {output_path} — Skipping.")
                continue

            # Load model/tokenizers as needed
            model = load_model(model_name, device)

            predictions = {}

            for idx, item in tqdm(doc_pairs.items(), desc=f"{dataset_name}-{model_name}"):
                if model_name == "BM25":
                    links = calculate_bm25_scores(dataset_name, item, corpus, nlp)

                elif model_name == "splade":
                    tokenizer, model_obj = model
                    links = calculate_splade_scores(dataset_name, item, corpus, model_obj, tokenizer, device)

                elif model_name in ["all-mpnet", "SFR"]:
                    instruction = "Instruct: Retrieve semantically similar text\nQuery: " if model_name == "SFR" else None
                    links = calculate_dense_scores(dataset_name, item, corpus, model, instruction)

                elif model_name == "bgem3-dense":
                    links = calculate_bgem3_dense_scores(dataset_name, item, corpus, model)

                elif model_name == "bgem3-sparse":
                    links = calculate_bgem3_sparse_scores(dataset_name, item, corpus, model)

                elif model_name == "dragon_plus":
                    tokenizer, model_obj = model
                    links = calculate_dense_scores(dataset_name, item, corpus, model_obj, tokenizer, device)
                

                elif model_name == "contriever":
                    tokenizer, model_obj = model
                    links = calculate_contriever_similarity(dataset_name, item, corpus, model_obj, tokenizer, device)

                elif model_type == "cross-encoder":
                    doc_texts = list(corpus[item["doc2"]].values())
                    doc_ids = list(corpus[item["doc2"]].keys())
                    links = {}
                    for source in item["links"].keys():
                        query = corpus[item["doc1"]][source]
                        pairs = [(query, doc) for doc in doc_texts]
                        scores = model.predict(pairs, batch_size=len(pairs))
                        result = {
                            doc_id: round(float(score), 6)
                            for doc_id, score in zip(doc_ids, scores)
                        }
                        links[source] = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

                predictions[idx] = {"links": links}

                # Periodic backup
                if len(predictions) % 100 == 0:
                    with open(output_path, "w") as f:
                        json.dump(predictions, f, indent=4)

            # Final save
            with open(output_path, "w") as f:
                json.dump(predictions, f, indent=4)

        elapsed = time.time() - start_time
        print(f"Completed {dataset_name} in {elapsed:.2f} seconds.")
        print("=" * 50)