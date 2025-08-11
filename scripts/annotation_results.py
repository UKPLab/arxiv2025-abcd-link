import json
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt


def stats(df, dataset_name="news"):
    # 1. Count of Link/No Link per annotator overall
    overall_counts = {
        "annotator_1": df["annotator_1"].value_counts(),
        "annotator_2": df["annotator_2"].value_counts(),
    }
    overall_counts = pd.DataFrame(overall_counts)

    # 2. Count of Link/No Link per annotator per method
    method_counts = (
        df.groupby("method")[["annotator_1", "annotator_2"]]
        .apply(lambda g: g.apply(pd.Series.value_counts))
        .fillna(0)
        .astype(int)
    )

    # 3. Agreement overall
    df["agreement"] = df["annotator_1"] == df["annotator_2"]

    # 4. Cohen's Kappa overall
    kappa_overall = cohen_kappa_score(df["annotator_1"], df["annotator_2"])
    # 5. Cohen's Kappa per method
    kappa_per_method = df.groupby("method", group_keys=False)[
        ["annotator_1", "annotator_2"]
    ].apply(lambda g: cohen_kappa_score(g["annotator_1"], g["annotator_2"]))

    # 6. Both annotators say "Link" per method
    df["both_link"] = (df["annotator_1"] == "Link") & (df["annotator_2"] == "Link")

    # Link rates individually
    both_link_rate = df.groupby("method")["both_link"].mean()
    both_link_rate_out = both_link_rate.rename({
        "llm": "LLM",
        "retriever": "Retriever",
        "mutual": "Overlap",
        "random": "Random"
    })

    # Aggregated link rates
    llm_overlap = df[df["method"].isin(["llm", "overlap"])]
    retriever_overlap = df[df["method"].isin(["retriever", "overlap"])]
    retriever_llm_union = df[df["method"].isin(["retriever", "llm", "overlap"])]

    llm_overlap_link_rate = llm_overlap["both_link"].mean()
    retriever_overlap_link_rate = retriever_overlap["both_link"].mean()
    retriever_llm_union_link_rate = retriever_llm_union["both_link"].mean()

    with open(f"./datasets/{dataset_name}_he/link_acceptance_rates.json", "w") as f:
        json.dump(both_link_rate_out.apply(lambda x: f"{x:.2%}").to_dict(), f, indent=4)
    print(f"Saved link acceptance rates under ./datasets/{dataset_name}_he/link_acceptance_rates.json")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # --- Plot 1: Overall Link/No Link counts ---
    overall_counts.plot(
        kind="bar",
        ax=axes[0, 0],
        edgecolor="black",
        zorder=3,
        color=["tab:purple", "tab:gray"],
    )
    axes[0, 0].set_title("Overall Link/No Link Counts per Annotator")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_xticklabels(overall_counts.index, rotation=0)
    axes[0, 0].grid(True, alpha=0.5, zorder=0)

    # --- Plot 2: Agreement per Method ---
    kappa_per_method.plot(
        kind="bar",
        ax=axes[0, 1],
        edgecolor="black",
        zorder=3,
        color=["tab:blue", "tab:orange", "tab:green", "tab:red"],
    )
    axes[0, 1].axhline(
        y=kappa_overall,
        color="black",
        linestyle="--",
        label="Overall Kappa",
        zorder=4,
    )
    axes[0, 1].set_title("Cohen's kappa per Method")
    axes[0, 1].set_ylabel("Kappa")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].set_xticklabels(["R+LLM", "Overlap", "Random", "Dragon+"], rotation=0)
    axes[0, 1].set_xlabel("")
    axes[0, 1].legend(loc="upper left")
    axes[0, 1].grid(True, alpha=0.5, zorder=0)

    # --- Plot 3: Both Annotators Saying 'Link' per Method ---
    combined_link_rate = both_link_rate.copy()
    combined_link_rate["retriever+llm"] = retriever_llm_union_link_rate
    combined_link_rate.plot(
        kind="bar",
        ax=axes[1, 0],
        edgecolor="black",
        zorder=3,
        color=["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:olive"],
    )
    axes[1, 0].set_title("Both Annotators Saying 'Link' per Method (i.e. Precision)")
    axes[1, 0].set_ylabel("Rate")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_xticklabels(["R+LLM", "Overlap", "Random", "Dragon+", "R+LLM &\nDragon+"], rotation=0)
    axes[1, 0].set_xlabel("")
    axes[1, 0].grid(True, alpha=0.5, zorder=0)

    # --- Plot 4: Both Annotators Saying 'Link' for Aggregated Methods ---
    agg_rates = pd.Series(
        {
            "R+LLM-only &\nOverlap": llm_overlap_link_rate,
            "Dragon+ &\nOverlap": retriever_overlap_link_rate,
            "R+LLM &\nDragon+": retriever_llm_union_link_rate,
        }
    )
    agg_rates.plot(
        kind="bar",
        ax=axes[1, 1],
        edgecolor="black",
        zorder=3,
        color=["tab:blue", "tab:red", "tab:olive"],
    )
    axes[1, 1].set_title("Both Annotators Saying 'Link' (Aggregated)")
    axes[1, 1].set_ylabel("Rate")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticklabels(agg_rates.index, rotation=0)
    axes[1, 1].grid(True, alpha=0.5, zorder=0)
    # Adjust layout
    plt.suptitle(f"Annotation Statistics ({dataset_name})")
    plt.tight_layout()
    plt.savefig(f"./datasets/{dataset_name}_he/plots.pdf", format="pdf", bbox_inches="tight")
    print(f"Saved plots under ./datasets/{dataset_name}_he/plots.pdf")


if __name__ == "__main__":
    news_annotations = pd.read_csv('./datasets/news_he/annotations.csv')
    news_annotations = news_annotations.reset_index(drop=True)
    stats(news_annotations, dataset_name="news")

    reviews_annotations = pd.read_csv('./datasets/reviews_he/annotations.csv')
    reviews_annotations = reviews_annotations.reset_index(drop=True)
    stats(reviews_annotations, dataset_name="reviews")