from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
import torch
import eval_ssl_fewshot as ssl_fewshot
from moabb import set_download_dir
from moabb.datasets import BNCI2014_001, PhysionetMI
from moabb.paradigms import LeftRightImagery

from eval_ssl_fewshot import (
    SubjectData,
    apply_standardizer,
    count_parameters,
    evaluate_subject,
    fit_standardizer,
    pretrain_ssl_encoder,
    set_seed,
)


def load_bnci_train(dataset: BNCI2014_001, paradigm: LeftRightImagery) -> np.ndarray:
    x_parts = []
    for subject in dataset.subject_list:
        x, _, m = paradigm.get_data(dataset=dataset, subjects=[subject])
        train_mask = (m["session"] == "0train").to_numpy()
        x_parts.append(x[train_mask])
    return np.concatenate(x_parts, axis=0)


def load_physio_subject(dataset: PhysionetMI, paradigm: LeftRightImagery, subject: int) -> SubjectData:
    x, y, m = paradigm.get_data(dataset=dataset, subjects=[subject])
    train_mask = m["run"].isin(["0", "1"]).to_numpy()
    test_mask = (m["run"] == "2").to_numpy()
    return SubjectData(x[train_mask], y[train_mask], x[test_mask], y[test_mask])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--csv", type=Path, default=Path("results_cross_bnci_to_physio.csv"))
    parser.add_argument("--weights", type=Path, default=Path("masked_ssl_bnci_pretrain.pt"))
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cache_dir = Path.cwd() / "moabb_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(cache_dir.resolve()))
    os.environ["MNE_DATA"] = str(cache_dir.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bnci = BNCI2014_001()
    physio = PhysionetMI()
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)

    x_bnci = load_bnci_train(bnci, paradigm)
    n_chans_bnci = x_bnci.shape[1]
    mean, std = fit_standardizer(x_bnci)
    x_bnci = apply_standardizer(x_bnci, mean, std)
    model = pretrain_ssl_encoder(
        x_pretrain=x_bnci,
        n_chans=x_bnci.shape[1],
        device=device,
        epochs=args.pretrain_epochs,
        batch_size=128,
        lr=1e-3,
        mask_ratio=0.3,
        seed=args.seed,
    )
    print(f"Encoder params: {count_parameters(model.encoder)}")
    print(f"Total params: {count_parameters(model)}")
    torch.save(model.encoder.state_dict(), args.weights)

    rows = []
    state = model.encoder.state_dict()
    physio_subjects = physio.subject_list[: args.subjects]
    min_trials_per_class = None
    for subject in physio_subjects:
        s = load_physio_subject(physio, paradigm, subject)
        current_min = min(np.sum(s.y_train == cls) for cls in np.unique(s.y_train))
        min_trials_per_class = (
            current_min if min_trials_per_class is None else min(min_trials_per_class, current_min)
        )
    feasible_tpc = tuple(t for t in ssl_fewshot.TRIALS_PER_CLASS if t <= int(min_trials_per_class))
    ssl_fewshot.TRIALS_PER_CLASS = feasible_tpc
    print(
        f"Using Physionet trials_per_class={feasible_tpc} "
        f"(min_available={min_trials_per_class})"
    )
    for subject in physio_subjects:
        s = load_physio_subject(physio, paradigm, subject)
        # Cross-dataset compatibility: BNCI pretraining uses 22 channels.
        # Physionet has 64 channels; use first 22 channels to match encoder input size.
        s = SubjectData(
            x_train=s.x_train[:, :n_chans_bnci, :],
            y_train=s.y_train,
            x_test=s.x_test[:, :n_chans_bnci, :],
            y_test=s.y_test,
        )
        s = SubjectData(
            x_train=apply_standardizer(s.x_train, mean, std),
            y_train=s.y_train,
            x_test=apply_standardizer(s.x_test, mean, std),
            y_test=s.y_test,
        )
        rows.extend(
            ssl_fewshot.evaluate_subject(
                subject=subject,
                subject_data=s,
                pretrained_encoder_state=state,
                device=device,
                finetune_epochs=args.finetune_epochs,
                finetune_lr=1e-3,
            )
        )
        print(f"Subject {subject}: done")
    pd.DataFrame(rows).to_csv(args.csv, index=False)
    print(f"Saved {args.csv}")


if __name__ == "__main__":
    main()
