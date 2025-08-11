import torch
import math


def cosine_similarity(vec1: dict, vec2: dict) -> float:
    common_keys = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
    norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = math.sqrt(sum(v**2 for v in vec2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

def splade_encode_batch(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits

    token_weights = torch.log(1 + torch.relu(logits))
    token_weights *= inputs["attention_mask"].unsqueeze(-1)

    batch_sparse = []
    for input_ids, weights in zip(inputs["input_ids"], token_weights):
        weights_max, _ = torch.max(weights, dim=-1)
        sparse_rep = {
            tokenizer.convert_ids_to_tokens(token_id.item()): weight.item()
            for token_id, weight in zip(input_ids, weights_max)
        }
        batch_sparse.append(sparse_rep)

    return batch_sparse
