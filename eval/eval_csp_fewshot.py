from pathlib import Path
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.decoding import CSP
from moabb import set_download_dir
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


TRIALS_PER_CLASS = (5, 10, 20, 50)
N_REPEATS = 5


def evaluate_subject(
    subject: int, dataset: BNCI2014_001, paradigm: LeftRightImagery
) -> list[dict[str, float | int]]:
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

    train_mask = metadata["session"] == "0train"
    test_mask = metadata["session"] == "1test"
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(
            f"Subject {subject} does not contain expected 0train/1test sessions."
        )

    X_train_all, y_train_all = X[train_mask.to_numpy()], y[train_mask.to_numpy()]
    X_test, y_test = X[test_mask.to_numpy()], y[test_mask.to_numpy()]

    classes = np.unique(y_train_all)
    if len(classes) != 2:
        raise RuntimeError(
            f"Subject {subject} expected 2 classes but found {len(classes)} classes."
        )

    class_indices = {cls: np.flatnonzero(y_train_all == cls) for cls in classes}
    for cls, idx in class_indices.items():
        if len(idx) < max(TRIALS_PER_CLASS):
            raise RuntimeError(
                f"Subject {subject}, class {cls} has only {len(idx)} training trials."
            )

    rows: list[dict[str, float | int]] = []
    for trials_per_class in TRIALS_PER_CLASS:
        repeat_accuracies: list[float] = []
        for repeat in range(N_REPEATS):
            rng = np.random.default_rng(seed=1000 * subject + 100 * trials_per_class + repeat)
            sampled_idx_parts = []
            for cls in classes:
                idx = class_indices[cls]
                sampled = rng.choice(idx, size=trials_per_class, replace=False)
                sampled_idx_parts.append(sampled)
            sampled_idx = np.concatenate(sampled_idx_parts)
            rng.shuffle(sampled_idx)

            X_cal, y_cal = X_train_all[sampled_idx], y_train_all[sampled_idx]

            pipeline = make_pipeline(
                CSP(n_components=8, log=True, norm_trace=False),
                LinearDiscriminantAnalysis(),
            )
            pipeline.fit(X_cal, y_cal)
            y_pred = pipeline.predict(X_test)
            repeat_accuracies.append(accuracy_score(y_test, y_pred))

        rows.append(
            {
                "subject": subject,
                "trials_per_class": trials_per_class,
                "mean_accuracy": float(np.mean(repeat_accuracies)),
                "std_accuracy": float(np.std(repeat_accuracies, ddof=0)),
            }
        )

    return rows


def plot_curves(results: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 6))
    for subject, group in results.groupby("subject"):
        group = group.sort_values("trials_per_class")
        plt.plot(
            group["trials_per_class"],
            group["mean_accuracy"],
            marker="o",
            linewidth=1.5,
            label=f"S{subject}",
        )
    plt.xlabel("Trials per class (calibration set from 0train)")
    plt.ylabel("Accuracy on full 1test")
    plt.title("BNCI2014_001 CSP+LDA Few-Shot Calibration Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CSP+LDA few-shot calibration on BNCI2014_001."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results_csp_fewshot.csv"),
        help="Path to save few-shot per-subject results as CSV.",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("results_csp_fewshot.png"),
        help="Path to save calibration curves plot.",
    )
    args = parser.parse_args()

    cache_dir = Path.cwd() / "moabb_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(cache_dir.resolve()))
    os.environ["MNE_DATA"] = str(cache_dir.resolve())

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)

    print(f"Using cache directory: {cache_dir.resolve()}")
    print("Evaluating BNCI2014_001 few-shot calibration with session split 0train -> 1test")
    print("Trials per class:", ", ".join(str(v) for v in TRIALS_PER_CLASS))
    print(f"Repeats per point: {N_REPEATS}")

    rows: list[dict[str, float | int]] = []
    for subject in dataset.subject_list:
        subject_rows = evaluate_subject(subject=subject, dataset=dataset, paradigm=paradigm)
        rows.extend(subject_rows)
        metrics = ", ".join(
            f"{int(r['trials_per_class'])}tpc={float(r['mean_accuracy']):.4f}+/-{float(r['std_accuracy']):.4f}"
            for r in subject_rows
        )
        print(f"Subject {subject:>2}: {metrics}")

    results = pd.DataFrame(rows).sort_values(["subject", "trials_per_class"]).reset_index(
        drop=True
    )
    summary = (
        results.groupby("trials_per_class")["mean_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_accuracy", "std": "std_across_subjects"})
    )

    print("-" * 70)
    print("Across-subject summary by trials_per_class")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.csv, index=False)
    print(f"Saved few-shot results to: {args.csv.resolve()}")

    plot_curves(results, args.plot)
    print(f"Saved calibration plot to: {args.plot.resolve()}")


if __name__ == "__main__":
    main()
