import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from retrieval.utils import splade_encode_batch, cosine_similarity


def calculate_bm25_scores(dataset, item, corpus, nlp):
    def tokenize_sentence(sentence):
        return [token.text for token in nlp(sentence)]

    links = {}
    docs = corpus[item["doc2"]]
    docs = [tokenize_sentence(doc) for doc in docs.values()]
    bm25 = BM25Okapi(docs)

    for source in item["links"]:
        query = tokenize_sentence(corpus[item["doc1"]][source])
        scores = bm25.get_scores(query)
        scores = {i: round(score, 6) for i, score in enumerate(scores)}
        links[source] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    return links


def calculate_dense_scores(dataset, item, corpus, model, instruction=None):
    links = {}
    docs = corpus[item["doc2"]]
    doc_texts = list(docs.values())
    target_embeddings = model.encode(doc_texts, convert_to_tensor=True)

    source_keys = list(item["links"].keys())
    queries = [
        f"{instruction or ''}{corpus[item['doc1']][source]}"
        for source in source_keys
    ]
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    scores_matrix = util.pytorch_cos_sim(query_embeddings, target_embeddings)

    for i, source in enumerate(source_keys):
        scores = {j: round(score, 6) for j, score in enumerate(scores_matrix[i].tolist())}
        links[source] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return links


def calculate_bgem3_dense_scores(dataset, item, corpus, model):
    links = {}
    docs = corpus[item["doc2"]]
    doc_texts = list(docs.values())
    target_embeddings = model.encode(
        doc_texts, return_dense=True, return_sparse=False,
        return_colbert_vecs=False, max_length=1024
    )["dense_vecs"]

    source_keys = list(item["links"].keys())
    queries = [corpus[item["doc1"]][source] for source in source_keys]
    query_embeddings = model.encode(
        queries, return_dense=True, return_sparse=False,
        return_colbert_vecs=False, max_length=1024
    )["dense_vecs"]

    scores_matrix = np.dot(query_embeddings, target_embeddings.T)
    for i, source in enumerate(source_keys):
        scores = {
            doc_id: round(float(score), 6)
            for doc_id, score in zip(docs.keys(), scores_matrix[i])
        }
        links[source] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return links


def calculate_bgem3_sparse_scores(dataset, item, corpus, model):
    links = {}
    docs = corpus[item["doc2"]]
    doc_texts = list(docs.values())
    target_sparse = model.encode(
        doc_texts, return_dense=False, return_sparse=True,
        return_colbert_vecs=False, max_length=1024
    )["lexical_weights"]

    source_keys = list(item["links"].keys())
    queries = [corpus[item["doc1"]][source] for source in source_keys]
    query_sparse_list = model.encode(
        queries, return_dense=False, return_sparse=True,
        return_colbert_vecs=False, max_length=1024
    )["lexical_weights"]

    for source, query_sparse in zip(source_keys, query_sparse_list):
        scores = {
            doc_id: float(model.compute_lexical_matching_score(query_sparse, doc_sparse))
            for doc_id, doc_sparse in zip(docs.keys(), target_sparse)
        }
        scores = {k: round(v, 6) for k, v in scores.items()}
        links[source] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return links


def calculate_contriever_similarity(dataset, item, corpus, model, tokenizer, device):
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]

    links = {}
    docs = corpus[item["doc2"]]
    doc_texts = list(docs.values())
    doc_inputs = tokenizer(doc_texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        doc_outputs = model(**doc_inputs)
    doc_embeddings = mean_pooling(doc_outputs.last_hidden_state, doc_inputs["attention_mask"])
    doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

    source_keys = list(item["links"].keys())
    queries = [corpus[item["doc1"]][source] for source in source_keys]
    query_inputs = tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_outputs = model(**query_inputs)
    query_embeddings = mean_pooling(query_outputs.last_hidden_state, query_inputs["attention_mask"])
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

    scores_matrix = torch.matmul(query_embeddings, doc_embeddings.T)
    for i, source in enumerate(source_keys):
        scores = {
            doc_id: round(float(score.item()), 6)
            for doc_id, score in zip(docs.keys(), scores_matrix[i])
        }
        links[source] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
    return links

def calculate_splade_scores(dataset, item, corpus, model, tokenizer, device):
    links = {}
    docs = corpus[item["doc2"]]
    doc_ids = list(docs.keys())
    doc_texts = list(docs.values())

    target_embeddings = splade_encode_batch(doc_texts, model, tokenizer, device)

    source_keys = list(item["links"].keys())
    queries = [corpus[item["doc1"]][source] for source in source_keys]
    query_embeddings = splade_encode_batch(queries, model, tokenizer, device)

    for source, query_embedding in zip(source_keys, query_embeddings):
        scores = {
            doc_id: cosine_similarity(query_embedding, doc_sparse)
            for doc_id, doc_sparse in zip(doc_ids, target_embeddings)
        }
        scores = {k: round(v, 6) for k, v in scores.items()}
        links[source] = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    return links