from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def normalize_method(mode: str) -> str:
    m = str(mode).lower()
    if m == "csp_lda":
        return "CSP+LDA"
    if "eegnet" in m:
        return "EEGNet (scratch)"
    if "mu_beta" in m:
        return "mu/beta SSL (FT)"
    if "ssl_full_finetune" in m:
        return "Masked SSL (FT)"
    return mode


def main() -> None:
    df = pd.read_csv("results_3datasets.csv")
    df = df.dropna(subset=["dataset", "trials_per_class", "mean_accuracy", "mode"]).copy()
    df["method"] = df["mode"].map(normalize_method)

    keep = {"CSP+LDA", "EEGNet (scratch)", "Masked SSL (FT)", "mu/beta SSL (FT)"}
    df = df[df["method"].isin(keep)]
    if df.empty:
        raise RuntimeError("No matching methods found in results_3datasets.csv")

    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        d = df[df["dataset"] == dataset]
        agg = (
            d.groupby(["method", "trials_per_class"])["mean_accuracy"]
            .mean()
            .reset_index()
            .sort_values("trials_per_class")
        )
        for method in sorted(agg["method"].unique()):
            m = agg[agg["method"] == method]
            ax.plot(
                m["trials_per_class"],
                m["mean_accuracy"] * 100.0,
                marker="o",
                linewidth=2,
                label=method,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Trials per class")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(alpha=0.3)
        ax.set_ylim(30, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = Path("calibration_curves_3datasets.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
