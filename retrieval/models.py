import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from FlagEmbedding import BGEM3FlagModel, FlagReranker


def load_model(model_name: str, device: str):
    if model_name == "BM25":
        return None
    elif model_name == "all-mpnet":
        return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    elif model_name == "SFR":
        return SentenceTransformer("Salesforce/SFR-Embedding-Mistral")
    elif model_name == "bgem3-dense" or model_name == "bgem3-sparse":
        return BGEM3FlagModel("BAAI/bge-m3", devices=device)
    elif model_name == "dragon_plus":
        tokenizer = AutoTokenizer.from_pretrained("facebook/dragon-plus-query-encoder")
        query_encoder = AutoModel.from_pretrained("facebook/dragon-plus-query-encoder").to(device).eval()
        context_encoder = AutoModel.from_pretrained("facebook/dragon-plus-context-encoder").to(device).eval()
        return tokenizer, query_encoder, context_encoder
    elif model_name == "splade":
        tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3")
        model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3").eval().to(device)
        return tokenizer, model
    elif model_name == "contriever":
        tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
        model = AutoModel.from_pretrained("facebook/contriever").eval().to(device)
        return tokenizer, model
    elif model_name == "ms_marco_MiniLM":
        return CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L6-v2",
            device=device,
            default_activation_function=torch.nn.Sigmoid(),
        )
    elif model_name == "bge-reranker":
        return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True, devices=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")