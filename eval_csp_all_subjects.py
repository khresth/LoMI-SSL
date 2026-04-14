from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
from mne.decoding import CSP
from moabb import set_download_dir
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline


def evaluate_subject(subject: int, dataset: BNCI2014_001, paradigm: LeftRightImagery) -> float:
    X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

    train_mask = metadata["session"] == "0train"
    test_mask = metadata["session"] == "1test"
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(
            f"Subject {subject} does not contain expected 0train/1test sessions."
        )

    X_train, y_train = X[train_mask.to_numpy()], y[train_mask.to_numpy()]
    X_test, y_test = X[test_mask.to_numpy()], y[test_mask.to_numpy()]

    pipeline = make_pipeline(
        CSP(n_components=8, log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return accuracy_score(y_test, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate CSP+LDA baseline on all BNCI2014_001 subjects."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save per-subject accuracies as CSV.",
    )
    args = parser.parse_args()

    cache_dir = Path.cwd() / "moabb_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(cache_dir.resolve()))
    os.environ["MNE_DATA"] = str(cache_dir.resolve())

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)

    print(f"Using cache directory: {cache_dir.resolve()}")
    print("Evaluating BNCI2014_001 with session split 0train -> 1test")

    rows: list[dict[str, float | int]] = []
    for subject in dataset.subject_list:
        acc = evaluate_subject(subject=subject, dataset=dataset, paradigm=paradigm)
        rows.append({"subject": subject, "accuracy": float(acc)})
        print(f"Subject {subject:>2}: accuracy={acc:.4f}")

    results = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    mean_acc = float(results["accuracy"].mean())
    std_acc = float(results["accuracy"].std(ddof=0))

    print("-" * 44)
    print(f"Mean accuracy: {mean_acc:.4f}")
    print(f"Std accuracy : {std_acc:.4f}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.csv, index=False)
        print(f"Saved per-subject results to: {args.csv.resolve()}")


if __name__ == "__main__":
    main()
