from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
import torch
import eval_ssl_fewshot as ssl_fewshot
from moabb import set_download_dir
from moabb.datasets import BNCI2014_001, BNCI2014_004, PhysionetMI
from moabb.paradigms import LeftRightImagery

from eval_ssl_fewshot import (
    SubjectData,
    apply_standardizer,
    count_parameters,
    fit_standardizer,
    pretrain_ssl_encoder,
    set_seed,
)


def load_subject_data(
    subject: int, dataset_name: str, dataset: object, paradigm: LeftRightImagery
) -> SubjectData:
    x, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    if dataset_name == "physionetmi":
        train_mask = metadata["run"].isin(["0", "1"])
        test_mask = metadata["run"] == "2"
    else:
        train_mask = metadata["session"] == "0train"
        test_mask = metadata["session"] == "1test"
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(f"Subject {subject} missing expected split for {dataset_name}.")
    return SubjectData(
        x_train=x[train_mask.to_numpy()],
        y_train=y[train_mask.to_numpy()],
        x_test=x[test_mask.to_numpy()],
        y_test=y[test_mask.to_numpy()],
    )


def make_dataset(dataset_name: str):
    if dataset_name == "bnci2014_001":
        return BNCI2014_001(), Path("results_masked_ssl_bnci2014_001.csv"), 128.0
    if dataset_name == "bnci2014_004":
        return BNCI2014_004(), Path("results_masked_ssl_bnci2014_004.csv"), 128.0
    if dataset_name == "physionetmi":
        return PhysionetMI(), Path("results_masked_ssl_physionetmi.csv"), 128.0
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["bnci2014_001", "bnci2014_004", "physionetmi"],
        required=True,
    )
    parser.add_argument("--subjects", type=int, default=0, help="0 means all.")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=Path("masked_ssl_encoder.pt"))
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trials-per-class",
        type=int,
        nargs="+",
        default=[5, 10, 20, 50],
        help="Requested few-shot budgets per class.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path.cwd() / "moabb_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(cache_dir.resolve()))
    os.environ["MNE_DATA"] = str(cache_dir.resolve())

    dataset, default_csv, _ = make_dataset(args.dataset)
    out_csv = args.csv or default_csv
    subject_list = dataset.subject_list if args.subjects <= 0 else dataset.subject_list[: args.subjects]
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)

    subject_data: dict[int, SubjectData] = {}
    x_pretrain_parts: list[np.ndarray] = []
    for subject in subject_list:
        sdata = load_subject_data(subject, args.dataset, dataset, paradigm)
        subject_data[subject] = sdata
        x_pretrain_parts.append(sdata.x_train)

    # Some datasets/splits (e.g., Physionet run-based split) do not have >=50 trials/class.
    min_trials_per_class = min(
        min(np.sum(s.y_train == cls) for cls in np.unique(s.y_train))
        for s in subject_data.values()
    )
    feasible_tpc = tuple(t for t in args.trials_per_class if t <= min_trials_per_class)
    if not feasible_tpc:
        raise RuntimeError(
            f"No requested trials_per_class fit available data (min per class={min_trials_per_class})."
        )
    ssl_fewshot.TRIALS_PER_CLASS = feasible_tpc
    print(
        f"Using trials_per_class={feasible_tpc} "
        f"(requested={tuple(args.trials_per_class)}, min_available={min_trials_per_class})"
    )

    x_pretrain_raw = np.concatenate(x_pretrain_parts, axis=0)
    mean, std = fit_standardizer(x_pretrain_raw)
    x_pretrain = apply_standardizer(x_pretrain_raw, mean, std)
    for subject in subject_list:
        s = subject_data[subject]
        subject_data[subject] = SubjectData(
            x_train=apply_standardizer(s.x_train, mean, std),
            y_train=s.y_train,
            x_test=apply_standardizer(s.x_test, mean, std),
            y_test=s.y_test,
        )

    ssl_model = pretrain_ssl_encoder(
        x_pretrain=x_pretrain,
        n_chans=x_pretrain.shape[1],
        device=device,
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        mask_ratio=args.mask_ratio,
        seed=args.seed,
    )
    print(f"Encoder params: {count_parameters(ssl_model.encoder)}")
    print(f"Total params: {count_parameters(ssl_model)}")
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ssl_model.encoder.state_dict(), args.weights)

    rows: list[dict[str, float | int | str]] = []
    state = ssl_model.encoder.state_dict()
    for subject in subject_list:
        rows.extend(
            ssl_fewshot.evaluate_subject(
                subject=subject,
                subject_data=subject_data[subject],
                pretrained_encoder_state=state,
                device=device,
                finetune_epochs=args.finetune_epochs,
                finetune_lr=args.finetune_lr,
            )
        )
        print(f"Subject {subject}: done")

    pd.DataFrame(rows).sort_values(["subject", "trials_per_class", "mode"]).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


if __name__ == "__main__":
    main()
