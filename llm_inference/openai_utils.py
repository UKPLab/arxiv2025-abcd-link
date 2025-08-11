from openai import OpenAI
from pydantic import BaseModel
from typing import List
import os


class Relation(BaseModel):
    sentID: int
    related: bool


class SentenceRelations(BaseModel):
    relations: List[Relation]


def get_openai_system_message(prompt_setup: str) -> str:
    if prompt_setup == "listwise":
        return (
            "You are an AI assistant specialized in evaluating sentence relations. "
            "You will get two related documents, along with a sentence from Document 1 (source) "
            "and a list of sentences from Document 2 (targets). The targets are ranked based on "
            "their similarity to the source sentence. Your task is to determine for each target "
            "sentence if it is related to the source sentence. This will help filter out irrelevant "
            "sentences and improve the quality of ranked sentences."
        )
    elif prompt_setup == "pairwise":
        return (
            "You are an AI assistant specialized in evaluating sentence relations. "
            "You will get two related documents, along with a sentence from Document 1 (source) "
            "and a sentence from Document 2 (target). Your task is to determine if the target sentence is related to the source sentence."
        )
    elif prompt_setup == "classification":
        return (
            "You are an AI assistant specialized in evaluating sentence relations. "
            "You will get two related documents, along with a sentence from Document 1 (source). "
            "Your task is to determine for each sentence in the target document if it is related to the source sentence. "
            "This will help filter out irrelevant sentences."
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_setup}")


def init_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    return OpenAI(api_key=api_key)
