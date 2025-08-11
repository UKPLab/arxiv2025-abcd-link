import os
import json

from llm_inference.prompts_loader import load_prompts, infer_prompt_setup
from llm_inference.executor import classify_dataset_batched_global
from llm_inference.chat_utils import load_llm
from llm_inference.openai_utils import init_openai_client
from llm_inference.openai_executor import classify_dataset_openai

PROMPT_DIR = "./data/prompts_json"
OUTPUT_DIR = "./llm_results"

# Supported models
MODELS = [
    "Qwen/Qwen2.5-32B-Instruct-AWQ",
    "microsoft/Phi-4",
    "gpt-4o"
]


def run_prompt_file(file_path: str, model_name: str):
    print(f"\n--- Running: {file_path} | Model: {model_name} ---")
    prompts = load_prompts(file_path)
    prompt_type = infer_prompt_setup(file_path)

    # Dispatch to appropriate LLM backend
    if model_name.startswith("gpt-"):
        client = init_openai_client()
        results = classify_dataset_openai(
            dataset_prompts=prompts,
            model_name=model_name,
            client=client,
            prompt_type=prompt_type,
        )
    else:
        tokenizer, llm = load_llm(model_name)
        results = classify_dataset_batched_global(
            dataset_prompts=prompts,
            model_name=model_name,
            llm=llm,
            tokenizer=tokenizer,
            prompt_type=prompt_type,
            batch_size=32
        )

    # Save results
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    model_tag = model_name.split("/")[-1]
    out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{prompt_type}_{model_tag}.json")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    prompt_files = sorted(
        fname for fname in os.listdir(PROMPT_DIR) if fname.endswith(".json")
    )

    for model in MODELS:
        for fname in prompt_files:
            run_prompt_file(os.path.join(PROMPT_DIR, fname), model)


