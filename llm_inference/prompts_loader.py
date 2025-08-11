import json
import os
from typing import Dict


def infer_prompt_setup(file_path: str) -> str:
    """
    Infers the prompt setup from the filename: pairwise, listwise, or classification.
    """
    file_name = os.path.basename(file_path)
    if "pairwise" in file_name:
        return "pairwise"
    elif "listwise" in file_name:
        return "listwise"
    elif "classification" in file_name:
        return "classification"
    else:
        raise ValueError(f"Cannot infer prompt setup from file name: {file_name}")


def load_prompts(file_path: str) -> Dict:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
