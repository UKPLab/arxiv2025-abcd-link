import json
import pandas as pd
import statistics
from collections import defaultdict


def load_gold_labels(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    result = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        doc_id = str(row['og_doc_id'])
        source_id = str(row['source_id'])
        target_id = int(row['target_id'])
        result[doc_id][source_id].append(target_id)
    return {doc_id: dict(source_map) for doc_id, source_map in result.items()}


def precision_score(y_true, y_pred):
    y_true_set = set(y_true)
    y_pred_set = set(y_pred)
    true_positives = len(y_true_set & y_pred_set)
    retrieved = len(y_pred_set)
    if retrieved == 0:
        return 0.0
    return true_positives / retrieved

def recall_score(y_true, y_pred):
    y_true_set = set(y_true)
    y_pred_set = set(y_pred)
    true_positives = len(y_true_set & y_pred_set)
    relevant = len(y_true_set)
    if relevant == 0:
        return 0.0
    return true_positives / relevant

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1


def evaluate_recall(recall_data, predictions, output_path):
    """
    Evaluate precision, recall, and F1 scores for each method.
    Save the results as JSON.
    """
    method_scores = {
        "R+LLM": {"precision": [], "recall": [], "f1": []},
        "Retriever": {"precision": [], "recall": [], "f1": []},
        "Random": {"precision": [], "recall": [], "f1": []},
    }

    for doc_id, source_targets in recall_data.items():
        for source_id, true_target_ids in source_targets.items():
            if doc_id not in predictions or source_id not in predictions[doc_id]:
                continue

            true_targets = [int(tid) for tid in true_target_ids]

            for method in method_scores.keys():
                predicted_target_ids = predictions[doc_id][source_id].get(method, [])
                predicted_targets = [int(tid) for tid in predicted_target_ids]

                try:
                    p, r, f1 = f1_score(true_targets, predicted_targets)
                except ZeroDivisionError:
                    p, r, f1 = 0.0, 0.0, 0.0

                method_scores[method]["precision"].append(p)
                method_scores[method]["recall"].append(r)
                method_scores[method]["f1"].append(f1)

    # Aggregate and save
    output_results = {}
    for method, scores in method_scores.items():
        avg_p = statistics.mean(scores["precision"]) if scores["precision"] else 0.0
        avg_r = statistics.mean(scores["recall"]) if scores["recall"] else 0.0
        avg_f1 = statistics.mean(scores["f1"]) if scores["f1"] else 0.0

        output_results[method] = {
            "precision": round(avg_p, 2),
            "recall": round(avg_r, 2),
            "f1": round(avg_f1, 2),
        }

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_results, f, indent=4)
    print(f"Saved true evaluation scores to {output_path}")


if __name__ == "__main__":
    news_gold = load_gold_labels("./datasets/news_he/news_gold_labels.csv")
    reviews_gold = load_gold_labels("./datasets/reviews_he/reviews_gold_labels.csv")

    # Load predictions
    with open("./datasets/news_he/predictions_news.json", "r") as f:
        predictions_news = json.load(f)

    with open("./datasets/reviews_he/predictions_reviews.json", "r") as f:
        predictions_reviews = json.load(f)

    # Evaluate and save results
    evaluate_recall(news_gold, predictions_news, "./datasets/news_he/eval_gold_labels.json")
    evaluate_recall(reviews_gold, predictions_reviews, "./datasets/reviews_he/eval_gold_labels.json")