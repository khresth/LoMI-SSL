from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    cross = pd.read_csv("results_cross_bnci_to_physio.csv")
    native = pd.read_csv("results_masked_ssl_physionetmi.csv")

    cross_ft = cross[cross["mode"] == "ssl_full_finetune"].copy()
    native_ft = native[native["mode"] == "ssl_full_finetune"].copy()

    c = (
        cross_ft.groupby("trials_per_class")["mean_accuracy"]
        .mean()
        .reset_index()
        .sort_values("trials_per_class")
    )
    n = (
        native_ft.groupby("trials_per_class")["mean_accuracy"]
        .mean()
        .reset_index()
        .sort_values("trials_per_class")
    )

    plt.figure(figsize=(6, 4))
    plt.plot(
        c["trials_per_class"],
        c["mean_accuracy"] * 100.0,
        marker="o",
        linewidth=2,
        label="BNCI-pretrained -> Physio FT",
    )
    plt.plot(
        n["trials_per_class"],
        n["mean_accuracy"] * 100.0,
        marker="o",
        linewidth=2,
        label="Physio-native masked SSL FT",
    )
    plt.xlabel("Trials per class")
    plt.ylabel("Accuracy (%)")
    plt.title("Cross-dataset transfer vs Physio-native")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    out = Path("cross_dataset_transfer.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
