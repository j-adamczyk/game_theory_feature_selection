import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

RESULTS_DIR = Path(__file__).parent.parent / "results"
PLOTS_DIR = Path(__file__).parent.parent / "plots"

BENCHMARKS = ["asap discovery", "expansionrx", "moleculenet", "moleculeace", "tdc"]
METHODS = [
    "none",
    "variance_threshold",
    "f_test_80",
    "mutual_info_80",
    "correlation_threshold",
    "boruta",
    "hsic_lasso",
    "permutation_importance",
    "rfecv",
    "select_from_model_l1",
    "shap",
    "sage",
    "shapley_effects",
]
METHOD_LABELS = {
    "none": "No selection",
    "variance_threshold": "Variance threshold",
    "f_test_80": "F-test (80%)",
    "mutual_info_80": "Mutual info (80%)",
    "correlation_threshold": "Corr. threshold (0.9)",
    "boruta": "Boruta",
    "hsic_lasso": "HSIC Lasso",
    "permutation_importance": "Permutation imp.",
    "rfecv": "RFECV",
    "select_from_model_l1": "L1 SelectFromModel",
    "shap": "SHAP (80%)",
    "sage": "SAGE (80%)",
    "shapley_effects": "Shapley Effects (80%)",
}
BENCHMARK_LABELS = {
    "asap discovery": "ASAP Discovery",
    "expansionrx": "ExpansionRx",
    "moleculenet": "MoleculeNet",
    "moleculeace": "MoleculeACE",
    "tdc": "TDC",
}


def load_all_results() -> pd.DataFrame:
    """Load all result CSVs into a single DataFrame."""
    frames = []
    for benchmark in BENCHMARKS:
        for method in METHODS:
            path = RESULTS_DIR / f"{benchmark}_{method}.csv"
            if not path.exists():
                print(f"Warning: {path} not found, skipping", file=sys.stderr)
                continue
            df = pd.read_csv(path)
            df["benchmark"] = benchmark
            df["method"] = method
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def get_available_methods(data: pd.DataFrame) -> list[str]:
    """Return METHODS filtered to only those present in the data."""
    present = set(data["method"].unique())
    return [m for m in METHODS if m in present]


def compute_ranks(data: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dataset ranks across methods.

    For regression (MAE), lower is better so rank ascending.
    For classification (AUROC), higher is better so rank descending.
    """
    rows = []
    for (benchmark, dataset), group in data.groupby(["benchmark", "dataset"]):
        task = group["task"].iloc[0]
        scores = group.set_index("method")["score"]
        if task == "regression":
            ranks = rankdata(scores.values, method="average")
        else:
            ranks = rankdata(-scores.values, method="average")
        for method, rank in zip(scores.index, ranks, strict=False):
            rows.append(
                {
                    "benchmark": benchmark,
                    "dataset": dataset,
                    "task": task,
                    "method": method,
                    "score": scores[method],
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def plot_avg_ranks(ranked: pd.DataFrame, methods: list[str]):
    """Bar chart of average rank per method: per benchmark + overall."""
    n_panels = len(BENCHMARKS) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5), sharey=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

    for idx, (key, title) in enumerate(
        list(BENCHMARK_LABELS.items()) + [("all", "Overall")]
    ):
        ax = axes[idx]
        subset = ranked if key == "all" else ranked[ranked["benchmark"] == key]
        avg_ranks = subset.groupby("method")["rank"].mean().reindex(methods)

        bars = ax.bar(
            range(len(methods)),
            avg_ranks.values,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            [METHOD_LABELS[m] for m in methods], rotation=45, ha="right", fontsize=8
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(1, len(methods))
        ax.axhline(
            y=np.mean(range(1, len(methods) + 1)),
            color="gray",
            linestyle="--",
            alpha=0.5,
        )
        for bar, val in zip(bars, avg_ranks.values, strict=False):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    axes[0].set_ylabel("Average rank (lower is better)", fontsize=11)
    fig.suptitle(
        "Average Rank of Feature Selection Methods", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "avg_rank.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_avg_scores(data: pd.DataFrame, methods: list[str]):
    """Bar chart of average metric per method: per benchmark + overall, split by task."""
    for task, metric in [("classification", "AUROC"), ("regression", "MAE")]:
        subset = data[data["task"] == task]
        panels = [
            (k, v)
            for k, v in BENCHMARK_LABELS.items()
            if len(subset[subset["benchmark"] == k]) > 0
        ]
        panels.append(("all", "Overall"))

        fig, axes = plt.subplots(
            1, len(panels), figsize=(5 * len(panels), 5), sharey=(task == "classification")
        )
        if len(panels) == 1:
            axes = [axes]
        colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))

        for idx, (key, title) in enumerate(panels):
            ax = axes[idx]
            bm_subset = subset if key == "all" else subset[subset["benchmark"] == key]
            means = bm_subset.groupby("method")["score"].mean().reindex(methods)

            bars = ax.bar(
                range(len(methods)),
                means.values,
                color=colors,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(
                [METHOD_LABELS[m] for m in methods], rotation=45, ha="right", fontsize=8
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            for bar, val in zip(bars, means.values, strict=False):
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                    )

        axes[0].set_ylabel(f"Average {metric}", fontsize=11)
        fig.suptitle(f"Average {metric} ({task})", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(
            PLOTS_DIR / f"avg_{metric.lower()}_{task}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def print_tsv_tables(data: pd.DataFrame, ranked: pd.DataFrame, methods: list[str]):
    """Print TSV tables for pasting into Excel."""
    print("=" * 80)
    print("TSV TABLES (copy-paste into Excel)")
    print("=" * 80)

    # Table 1: Average scores per benchmark and task
    print("\n--- Average Score per Benchmark x Method (split by task) ---")
    print("Benchmark\tTask\t" + "\t".join(METHOD_LABELS[m] for m in methods))
    for benchmark in BENCHMARKS:
        for task in ["classification", "regression"]:
            subset = data[(data["benchmark"] == benchmark) & (data["task"] == task)]
            if len(subset) == 0:
                continue
            values = []
            for method in methods:
                m_data = subset[subset["method"] == method]
                values.append(
                    f"{m_data['score'].mean():.4f}" if len(m_data) > 0 else ""
                )
            print(f"{BENCHMARK_LABELS[benchmark]}\t{task}\t" + "\t".join(values))

    # Overall averages
    for task in ["classification", "regression"]:
        subset = data[data["task"] == task]
        if len(subset) == 0:
            continue
        values = []
        for method in methods:
            m_data = subset[subset["method"] == method]
            values.append(f"{m_data['score'].mean():.4f}" if len(m_data) > 0 else "")
        print(f"Overall\t{task}\t" + "\t".join(values))

    # Table 2: Average ranks
    print("\n--- Average Rank per Benchmark x Method ---")
    print("Benchmark\t" + "\t".join(METHOD_LABELS[m] for m in methods))
    for benchmark in BENCHMARKS:
        subset = ranked[ranked["benchmark"] == benchmark]
        values = []
        for method in methods:
            m_data = subset[subset["method"] == method]
            values.append(f"{m_data['rank'].mean():.2f}" if len(m_data) > 0 else "")
        print(f"{BENCHMARK_LABELS[benchmark]}\t" + "\t".join(values))

    # Overall rank
    values = []
    for method in methods:
        m_data = ranked[ranked["method"] == method]
        values.append(f"{m_data['rank'].mean():.2f}" if len(m_data) > 0 else "")
    print("Overall\t" + "\t".join(values))

    # Table 3: Average ranks split by task
    print("\n--- Average Rank per Benchmark x Method (split by task) ---")
    print("Benchmark\tTask\t" + "\t".join(METHOD_LABELS[m] for m in methods))
    for benchmark in BENCHMARKS:
        for task in ["classification", "regression"]:
            subset = ranked[
                (ranked["benchmark"] == benchmark) & (ranked["task"] == task)
            ]
            if len(subset) == 0:
                continue
            values = []
            for method in methods:
                m_data = subset[subset["method"] == method]
                values.append(f"{m_data['rank'].mean():.2f}" if len(m_data) > 0 else "")
            print(f"{BENCHMARK_LABELS[benchmark]}\t{task}\t" + "\t".join(values))

    for task in ["classification", "regression"]:
        subset = ranked[ranked["task"] == task]
        if len(subset) == 0:
            continue
        values = []
        for method in methods:
            m_data = subset[subset["method"] == method]
            values.append(f"{m_data['rank'].mean():.2f}" if len(m_data) > 0 else "")
        print(f"Overall\t{task}\t" + "\t".join(values))

    # Table 4: Per-dataset scores
    print("\n--- Per-Dataset Scores ---")
    print("Benchmark\tDataset\tTask\t" + "\t".join(METHOD_LABELS[m] for m in methods))
    for benchmark in BENCHMARKS:
        bm_data = data[data["benchmark"] == benchmark]
        for dataset in bm_data["dataset"].unique():
            ds_data = bm_data[bm_data["dataset"] == dataset]
            task = ds_data["task"].iloc[0]
            values = []
            for method in methods:
                m_data = ds_data[ds_data["method"] == method]
                values.append(
                    f"{m_data['score'].values[0]:.4f}" if len(m_data) > 0 else ""
                )
            print(
                f"{BENCHMARK_LABELS[benchmark]}\t{dataset}\t{task}\t"
                + "\t".join(values)
            )

    # Table 5: Win counts
    print("\n--- Win Counts (how many datasets each method ranks #1) ---")
    print("Benchmark\tTask\t" + "\t".join(METHOD_LABELS[m] for m in methods))
    for benchmark in BENCHMARKS:
        for task in ["classification", "regression"]:
            subset = ranked[
                (ranked["benchmark"] == benchmark) & (ranked["task"] == task)
            ]
            if len(subset) == 0:
                continue
            values = []
            for method in methods:
                m_data = subset[subset["method"] == method]
                wins = (m_data["rank"] == 1).sum()
                values.append(str(wins))
            print(f"{BENCHMARK_LABELS[benchmark]}\t{task}\t" + "\t".join(values))

    for task in ["classification", "regression"]:
        subset = ranked[ranked["task"] == task]
        if len(subset) == 0:
            continue
        values = []
        for method in methods:
            m_data = subset[subset["method"] == method]
            wins = (m_data["rank"] == 1).sum()
            values.append(str(wins))
        print(f"Overall\t{task}\t" + "\t".join(values))


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    data = load_all_results()
    methods = get_available_methods(data)
    ranked = compute_ranks(data)

    plot_avg_ranks(ranked, methods)
    plot_avg_scores(data, methods)

    print_tsv_tables(data, ranked, methods)

    print(f"\nPlots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
