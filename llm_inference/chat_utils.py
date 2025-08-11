from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
from pydantic import BaseModel
from typing import List


# -----------------------------
# Pydantic Output Schema
# -----------------------------
class Relation(BaseModel):
    sentence_id: int
    related: bool


class SentenceRelations(BaseModel):
    relations: List[Relation]


def get_guided_sampling_params(schema: BaseModel, temperature: float = 0.4) -> SamplingParams:
    json_schema = schema.model_json_schema()
    guided = GuidedDecodingParams(json=json_schema)
    return SamplingParams(
        temperature=temperature,
        top_p=0.9,
        min_p=0.1,
        guided_decoding=guided,
    )


def load_llm(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(model=model_name)
    return tokenizer, llm


def build_chat_prompt(text: str, tokenizer, prompt_setup: str) -> str:
    """
    Builds a prompt using the appropriate system instruction based on the prompt setup.
    """

    if prompt_setup == "listwise":
        system_message = (
            "You are an AI assistant specialized in evaluating sentence relations.\n"
            "You will get two related documents, along with a sentence from Document 1 (source) "
            "and a list of sentences from Document 2 (targets). The targets are ranked based on their similarity "
            "to the source sentence. Your task is to determine for each target sentence if it is related to the source sentence. "
            "This will help filter out irrelevant sentences and improve the quality of the ranked sentences."
        )
    elif prompt_setup == "pairwise":
        system_message = (
            "You are an AI assistant specialized in evaluating sentence relations.\n"
            "You will get two related documents, along with a sentence from Document 1 (source) "
            "and a sentence from Document 2 (target). Your task is to determine if the target sentence is related to the source sentence."
        )
    elif prompt_setup == "classification":
        system_message = (
            "You are an AI assistant specialized in evaluating sentence relations.\n"
            "You will get two related documents, along with a sentence from Document 1 (source). "
            "Your task is to determine for each sentence in the target document if it is related to the source sentence. "
            "This will help filter out irrelevant sentences."
        )
    else:
        raise ValueError(f"Unknown prompt_setup: {prompt_setup}")

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": text},
    ]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
