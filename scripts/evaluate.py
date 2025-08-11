import os
import glob
import json
import argparse
import numpy as np


def precision_at_k(predicted, ground_truth, k=None):
    k = k or len(predicted)
    k = min(k, len(predicted))
    top_k = predicted[:k]
    true_positives = sum(1 for doc_id in top_k if doc_id in ground_truth)
    return true_positives / k if k > 0 else 0


def recall_at_k(predicted, ground_truth, k=None):
    k = k or len(predicted)
    top_k = predicted[:k]
    true_positives = sum(1 for doc_id in top_k if doc_id in ground_truth)
    return true_positives / len(ground_truth) if ground_truth else 0


def f1_at_k(precision, recall, beta=1):
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def evaluate_ranked_query(predicted_dict, ground_truth, cutoffs):
    predicted_sorted = sorted(predicted_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_ids = [doc_id for doc_id, _ in predicted_sorted]

    metrics = {}
    for k in cutoffs:
        p = precision_at_k(predicted_ids, ground_truth, k)
        r = recall_at_k(predicted_ids, ground_truth, k)
        f1 = f1_at_k(p, r)
        metrics[k] = {"precision": p, "recall": r, "f1": f1}
    return metrics


def evaluate_ranked_predictions(ground_truth_file, prediction_file, cutoffs):
    with open(ground_truth_file, "r") as f:
        gt_data = json.load(f)
    with open(prediction_file, "r") as f:
        pred_data = json.load(f)

    aggregated = {k: {"precision": [], "recall": [], "f1": []} for k in cutoffs}

    for idx in gt_data:
        if idx not in pred_data:
            continue
        gt_links = gt_data[idx]["links"]
        pred_links = pred_data[idx]["links"]

        for query_id, gt_list in gt_links.items():
            if query_id not in pred_links:
                continue
            predicted_dict = pred_links[query_id]
            query_metrics = evaluate_ranked_query(predicted_dict, gt_list, cutoffs)
            for k in cutoffs:
                for m in ["precision", "recall", "f1"]:
                    aggregated[k][m].append(query_metrics[k][m])

    averaged = {
        k: {
            "precision": round(np.mean(aggregated[k]["precision"]), 4),
            "recall": round(np.mean(aggregated[k]["recall"]), 4),
            "f1": round(np.mean(aggregated[k]["f1"]), 4),
        }
        for k in cutoffs
    }
    return averaged


def evaluate_classified_query(predicted_dict, ground_truth):
    predicted_ids = [doc_id for doc_id, v in predicted_dict.items() if v]
    p = precision_at_k(predicted_ids, ground_truth, k=None)
    r = recall_at_k(predicted_ids, ground_truth, k=None)
    f1 = f1_at_k(p, r)
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_classified_predictions(ground_truth_file, prediction_file):
    with open(ground_truth_file, "r") as f:
        gt_data = json.load(f)
    with open(prediction_file, "r") as f:
        pred_data = json.load(f)

    aggregated = {"precision": [], "recall": [], "f1": []}

    for idx in gt_data:
        if idx not in pred_data:
            continue
        gt_links = gt_data[idx]["links"]
        pred_links = pred_data[idx]["links"]

        for query_id, gt_list in gt_links.items():
            if query_id not in pred_links:
                continue
            predicted_dict = pred_links[query_id]
            query_metrics = evaluate_classified_query(predicted_dict, gt_list)
            for m in ["precision", "recall", "f1"]:
                aggregated[m].append(query_metrics[m])

    return {
        "precision": round(np.mean(aggregated["precision"]), 4),
        "recall": round(np.mean(aggregated["recall"]), 4),
        "f1": round(np.mean(aggregated["f1"]), 4),
    }


def calculate_metrics(prediction_type="ranked", cutoffs=None, metric=None):
    datasets = ["news_ecb", "news_synth", "reviews_synth", "reviews_f1000"]
    base_paths = {
        "ranked": "./predictions",
        "classified": "./llm_results",
    }

    if prediction_type == "ranked" and cutoffs is None:
        cutoffs = [1, 3, 5, 7, 10, 20]

    metrics = {ds: {} for ds in datasets}

    for dataset in datasets:
        gt_path = f"./datasets/{dataset}/{dataset}_links.json"

        if prediction_type == "ranked":
            pred_files = glob.glob(
                os.path.join(base_paths[prediction_type], dataset, "*.json")
            )
        else:
            pred_files = glob.glob(
                os.path.join(base_paths[prediction_type], f"{dataset}*.json")
            )

        pred_files = [f for f in pred_files]
        if not pred_files:
            print(f"No prediction files found for {dataset}. Skipping.")
            continue

        if prediction_type == "ranked":
            for k in cutoffs:
                metrics[dataset][k] = {}

            for file in pred_files:
                model_name = (
                    os.path.basename(file)
                    .replace(f"{dataset}_", "")
                    .split("_scores")[0]
                )
                model_metrics = evaluate_ranked_predictions(gt_path, file, cutoffs)
                for k in cutoffs:
                    metrics[dataset][k][model_name] = (
                        model_metrics[k][metric] if metric else model_metrics[k]
                    )

        elif prediction_type == "classified":
            for file in pred_files:
                model_name = (
                    os.path.basename(file)
                    .replace(f"{dataset}_", "")
                    .split("_scores")[0]
                )
                model_metrics = evaluate_classified_predictions(gt_path, file)
                metrics[dataset][model_name] = (
                    model_metrics[metric] if metric else model_metrics
                )

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument(
        "--type",
        choices=["ranked", "classified", "all"],
        default="all",
        help="Type of evaluation.",
    )
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 10, 20],
        help="Cutoffs for ranking evaluation.",
    )
    parser.add_argument(
        "--metric",
        choices=["precision", "recall", "f1"],
        default=None,
        help="Report only this metric.",
    )

    args = parser.parse_args()

    os.makedirs("./eval_outputs", exist_ok=True)

    if args.type in ["ranked", "all"]:
        print("Evaluating ranked predictions...")
        ranked = calculate_metrics(
            prediction_type="ranked", cutoffs=args.cutoffs, metric=args.metric
        )
        with open("./eval_outputs/ranked_metrics.json", "w") as f:
            json.dump(ranked, f, indent=4)
        print("Saved: eval_outputs/ranked_metrics.json")

    if args.type in ["classified", "all"]:
        print("Evaluating classified predictions...")
        classified = calculate_metrics(prediction_type="classified", metric=args.metric)
        with open("./eval_outputs/classified_metrics.json", "w") as f:
            json.dump(classified, f, indent=4)
        print("Saved: eval_outputs/classified_metrics.json")
