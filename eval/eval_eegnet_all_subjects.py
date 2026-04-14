from pathlib import Path
import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from braindecode.classifier import EEGClassifier
from braindecode.models import EEGNetv4
from moabb import set_download_dir
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


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

    X_train = X_train.astype(np.float32, copy=False)
    X_test = X_test.astype(np.float32, copy=False)

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train).astype(np.int64, copy=False)
    y_test_enc = encoder.transform(y_test).astype(np.int64, copy=False)

    n_chans = X_train.shape[1]
    n_times = X_train.shape[2]
    n_classes = len(encoder.classes_)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EEGNetv4(
        n_chans=n_chans,
        n_times=n_times,
        n_outputs=n_classes,
        final_conv_length="auto",
    )
    net = EEGClassifier(
        model,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__lr=1e-3,
        batch_size=64,
        max_epochs=100,
        train_split=None,
        iterator_train__shuffle=True,
        verbose=0,
        device=device,
    )
    net.fit(X_train, y_train_enc)
    y_pred = net.predict(X_test)
    return accuracy_score(y_test_enc, y_pred)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate EEGNet baseline on all BNCI2014_001 subjects."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to save per-subject accuracies as CSV.",
    )
    args = parser.parse_args()

    warnings.filterwarnings(
        "ignore",
        message=r".*EEGNetv4\(\) is a deprecated class.*",
        category=FutureWarning,
    )

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
