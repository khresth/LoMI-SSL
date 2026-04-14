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
from moabb.datasets import BNCI2014_001, PhysionetMI
from moabb.paradigms import LeftRightImagery
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


TRIALS_PER_CLASS = (5, 10, 20, 50)
N_REPEATS = 5


def sample_fewshot_indices(y_train: np.ndarray, trials_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    idxs = [rng.choice(np.flatnonzero(y_train == c), size=trials_per_class, replace=False) for c in classes]
    all_idx = np.concatenate(idxs)
    rng.shuffle(all_idx)
    return all_idx


def subject_split(dataset_name: str, metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if dataset_name == "physionetmi":
        train_mask = metadata["run"].isin(["0", "1"]).to_numpy()
        test_mask = (metadata["run"] == "2").to_numpy()
    else:
        train_mask = (metadata["session"] == "0train").to_numpy()
        test_mask = (metadata["session"] == "1test").to_numpy()
    return train_mask, test_mask


def eval_subject(dataset_name: str, dataset: object, paradigm: LeftRightImagery, subject: int) -> list[dict]:
    x, y, m = paradigm.get_data(dataset=dataset, subjects=[subject])
    tr_mask, te_mask = subject_split(dataset_name, m)
    x_tr, y_tr = x[tr_mask].astype(np.float32), y[tr_mask]
    x_te, y_te = x[te_mask].astype(np.float32), y[te_mask]
    le = LabelEncoder()
    y_te_enc = le.fit_transform(y_te)

    rows = []
    for tpc in TRIALS_PER_CLASS:
        accs = []
        for rep in range(N_REPEATS):
            idx = sample_fewshot_indices(y_tr, tpc, 1000 * subject + 100 * tpc + rep)
            x_cal, y_cal = x_tr[idx], y_tr[idx]
            y_cal_enc = le.transform(y_cal)
            model = EEGNetv4(
                n_chans=x_cal.shape[1],
                n_times=x_cal.shape[2],
                n_outputs=len(le.classes_),
                final_conv_length="auto",
            )
            net = EEGClassifier(
                model,
                criterion=torch.nn.CrossEntropyLoss,
                optimizer=torch.optim.Adam,
                optimizer__lr=1e-3,
                batch_size=64,
                max_epochs=80,
                train_split=None,
                iterator_train__shuffle=True,
                verbose=0,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            net.fit(x_cal, y_cal_enc.astype(np.int64))
            y_pred = net.predict(x_te)
            accs.append(accuracy_score(y_te_enc, y_pred))
        rows.append(
            {
                "subject": subject,
                "trials_per_class": tpc,
                "mode": "eegnet_from_scratch",
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs, ddof=0)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["bnci2014_001", "physionetmi"], required=True)
    parser.add_argument("--subjects", type=int, default=0)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=r".*EEGNetv4\(\) is a deprecated class.*")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cache_dir = Path.cwd() / "moabb_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(cache_dir.resolve()))
    os.environ["MNE_DATA"] = str(cache_dir.resolve())
    dataset = BNCI2014_001() if args.dataset == "bnci2014_001" else PhysionetMI()
    subjects = dataset.subject_list if args.subjects <= 0 else dataset.subject_list[: args.subjects]
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)

    rows = []
    for s in subjects:
        rows.extend(eval_subject(args.dataset, dataset, paradigm, s))
        print(f"Subject {s}: done")
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
