import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------

CSV_PATH = "faq_log.csv"                      # Input log from the generator
METRICS_CSV_PATH = "metrics_summary.csv"      # Output metrics table
SAMPLES_CSV_PATH = "faq_samples_for_rating.csv"  # Sampled FAQs for human rating

# Sampling configuration
SAMPLES_PER_DOC_PROVIDER_PROMPT = 5  # rows per (document, provider, prompt, model)
RANDOM_SEED = 42

# ----------------------------------------


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load the FAQ log CSV and add helper columns (like answer_length).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Basic sanity checks
    required_cols = ["document", "provider", "model", "prompt", "question", "answer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in CSV: {col}")

    # Add answer length (word count)
    df["answer"] = df["answer"].astype(str)
    df["answer_length"] = df["answer"].apply(lambda x: len(x.split()))

    return df


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute:
      - total_faqs
      - unique_questions
      - unique_ratio
      - avg_answer_length
    per (document, provider, model, prompt).
    """
    group_cols = ["document", "provider", "model", "prompt"]

    metrics = (
        df.groupby(group_cols)
        .agg(
            total_faqs=("question", "count"),
            unique_questions=("question", lambda x: x.str.lower().nunique()),
            avg_answer_length=("answer_length", "mean"),
        )
        .reset_index()
    )

    metrics["unique_ratio"] = metrics["unique_questions"] / metrics["total_faqs"]

    # Sort for readability
    metrics = metrics.sort_values(group_cols).reset_index(drop=True)

    return metrics


def save_metrics(metrics: pd.DataFrame, out_path: str) -> None:
    """
    Save metrics DataFrame to CSV.
    """
    metrics.to_csv(out_path, index=False)
    print(f"[+] Saved metrics to {out_path}")


# ------------- GROUPED PLOTS (OpenAI vs Gemini) -------------


def _prepare_grouped_pivot(metrics: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Helper: pivot metrics into a table with:
      index: document + prompt
      columns: provider (openai, gemini)
      values: value_col
    """
    # Combined label for x-axis categories
    metrics = metrics.copy()
    metrics["xp_label"] = metrics["document"] + "\n" + metrics["prompt"]

    pivot = metrics.pivot_table(
        index="xp_label",
        columns="provider",
        values=value_col,
        aggfunc="mean",
    ).fillna(0)

    # Ensure consistent column order
    providers = [c for c in ["openai", "gemini"] if c in pivot.columns]
    pivot = pivot[providers]

    return pivot


def plot_faq_counts_grouped(metrics: pd.DataFrame, out_path: str) -> None:
    """
    Double bar chart: FAQ count for OpenAI vs Gemini side-by-side.
    X-axis = (document, prompt)
    """
    pivot = _prepare_grouped_pivot(metrics, "total_faqs")

    ax = pivot.plot(
        kind="bar",
        figsize=(14, 6),
        color=["skyblue", "salmon"][: len(pivot.columns)],
        width=0.8,
    )

    plt.title("FAQ Count by Provider (OpenAI vs Gemini)")
    plt.ylabel("Total FAQs")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Provider")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_unique_ratio_grouped(metrics: pd.DataFrame, out_path: str) -> None:
    """
    Double bar chart: unique question ratio for OpenAI vs Gemini.
    """
    pivot = _prepare_grouped_pivot(metrics, "unique_ratio")

    ax = pivot.plot(
        kind="bar",
        figsize=(14, 6),
        color=["skyblue", "salmon"][: len(pivot.columns)],
        width=0.8,
    )

    plt.title("Unique Question Ratio by Provider (OpenAI vs Gemini)")
    plt.ylabel("Unique Ratio")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Provider")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_avg_answer_length_grouped(metrics: pd.DataFrame, out_path: str) -> None:
    """
    Double bar chart: average answer length (words) for OpenAI vs Gemini.
    """
    pivot = _prepare_grouped_pivot(metrics, "avg_answer_length")

    ax = pivot.plot(
        kind="bar",
        figsize=(14, 6),
        color=["skyblue", "salmon"][: len(pivot.columns)],
        width=0.8,
    )

    plt.title("Average Answer Length by Provider (OpenAI vs Gemini)")
    plt.ylabel("Avg Answer Length (words)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Provider")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved: {out_path}")


# ------------- SAMPLING FOR HUMAN EVALUATION -------------


def sample_for_human_evaluation(
    df: pd.DataFrame,
    samples_per_group: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample rows for human evaluation.

    Strategy:
      - Group by (document, provider, model, prompt)
      - Sample up to `samples_per_group` rows from each group
      - Add columns for human ratings:
          usefulness (1–5), clarity (1–5), correctness (1–5)

    Returns the sampled DataFrame.
    """
    group_cols = ["document", "provider", "model", "prompt"]

    def sample_group(group: pd.DataFrame) -> pd.DataFrame:
        n = min(samples_per_group, len(group))
        return group.sample(n=n, random_state=random_state)

    sampled = (
        df.groupby(group_cols, group_keys=False)
        .apply(sample_group)
        .reset_index(drop=True)
    )

    # Keep only relevant columns for rating
    sampled = sampled[
        ["document", "provider", "model", "prompt", "question", "answer"]
    ].copy()

    # Add blank columns for human rating
    sampled["usefulness_1_5"] = ""
    sampled["clarity_1_5"] = ""
    sampled["correctness_1_5"] = ""

    return sampled


# ------------- MAIN -------------


def main():
    print("[*] Loading FAQ log CSV...")
    df = load_data(CSV_PATH)
    print(f"[+] Loaded {len(df)} rows from {CSV_PATH}")

    # ---------------- METRICS ----------------
    print("[*] Computing metrics...")
    metrics = compute_metrics(df)
    print("[+] Metrics computed:")
    print(metrics)

    save_metrics(metrics, METRICS_CSV_PATH)

    # Create output directory for plots if needed
    out_dir = Path(".")
    counts_plot_path = out_dir / "faq_counts.png"
    ratio_plot_path = out_dir / "unique_ratio.png"
    length_plot_path = out_dir / "avg_answer_length.png"

    print("[*] Generating grouped plots (OpenAI vs Gemini)...")
    plot_faq_counts_grouped(metrics, str(counts_plot_path))
    plot_unique_ratio_grouped(metrics, str(ratio_plot_path))
    plot_avg_answer_length_grouped(metrics, str(length_plot_path))

    # ---------------- SAMPLING ----------------
    print("[*] Sampling rows for human evaluation...")
    sampled = sample_for_human_evaluation(
        df,
        samples_per_group=SAMPLES_PER_DOC_PROVIDER_PROMPT,
        random_state=RANDOM_SEED,
    )
    sampled.to_csv(SAMPLES_CSV_PATH, index=False)
    print(f"[+] Saved sampled rows for rating to {SAMPLES_CSV_PATH}")
    print("[*] Done.")


if __name__ == "__main__":
    main()
