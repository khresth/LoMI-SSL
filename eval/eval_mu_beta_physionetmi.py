from pathlib import Path
import argparse
import os

import numpy as np
import pandas as pd
import torch
from moabb import set_download_dir
from moabb.datasets import PhysionetMI
from moabb.paradigms import LeftRightImagery

from eval_mu_beta_ssl import (
    SubjectData,
    apply_standardizer,
    count_parameters,
    evaluate_subject,
    fit_standardizer,
    pretrain_mu_beta_ssl,
    set_seed,
)


def load_subject_data(subject: int, dataset: PhysionetMI, paradigm: LeftRightImagery) -> SubjectData:
    x, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])

  
    train_mask = metadata["run"].isin(["0", "1"])
    test_mask = metadata["run"] == "2"
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(
            f"Subject {subject} does not contain expected run split (0/1 -> 2)."
        )

    return SubjectData(
        x_train=x[train_mask.to_numpy()],
        y_train=y[train_mask.to_numpy()],
        x_test=x[test_mask.to_numpy()],
        y_test=y[test_mask.to_numpy()],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mu/beta SSL few-shot benchmark on PhysionetMI (first 20 subjects)."
    )
    parser.add_argument("--csv", type=Path, default=Path("results_mu_beta_physionetmi.csv"))
    parser.add_argument(
        "--weights", type=Path, default=Path("mu_beta_ssl_encoder_physionetmi_20subj.pt")
    )
    parser.add_argument("--pretrain-epochs", type=int, default=40)
    parser.add_argument("--pretrain-batch-size", type=int, default=128)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--finetune-epochs", type=int, default=80)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path.cwd() / "moabb_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    set_download_dir(str(cache_dir.resolve()))
    os.environ["MNE_DATA"] = str(cache_dir.resolve())

    dataset = PhysionetMI()
    subject_list = dataset.subject_list[:20]
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)
    sfreq = 128.0

    print(f"Using device: {device}")
    print(f"Using cache directory: {cache_dir.resolve()}")
    print("Dataset: PhysionetMI (subjects 1-20)")
    print("Split: runs 0/1 -> run 2")

    subject_data: dict[int, SubjectData] = {}
    x_pretrain_parts: list[np.ndarray] = []
    for subject in subject_list:
        sdata = load_subject_data(subject=subject, dataset=dataset, paradigm=paradigm)
        subject_data[subject] = sdata
        x_pretrain_parts.append(sdata.x_train)

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

    print("Pretraining compact mu/beta-contrastive encoder...")
    ssl_model = pretrain_mu_beta_ssl(
        x_full=x_pretrain,
        sfreq=sfreq,
        n_chans=x_pretrain.shape[1],
        device=device,
        epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        temperature=args.temperature,
        seed=args.seed,
    )

    encoder_params = count_parameters(ssl_model.encoder)
    total_params = count_parameters(ssl_model)
    print(f"Encoder params: {encoder_params}")
    print(f"Total SSL model params: {total_params}")
    if total_params >= 100_000:
        raise RuntimeError(f"Model too large ({total_params} params), expected <100k.")

    args.weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ssl_model.encoder.state_dict(), args.weights)
    print(f"Saved pretrained encoder weights to: {args.weights.resolve()}")

    rows: list[dict[str, float | int | str]] = []
    pretrained_encoder_state = ssl_model.encoder.state_dict()
    for subject in subject_list:
        subject_rows = evaluate_subject(
            subject=subject,
            subject_data=subject_data[subject],
            pretrained_encoder_state=pretrained_encoder_state,
            device=device,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
        )
        rows.extend(subject_rows)
        print(f"Subject {subject:>2}: completed")

    results = pd.DataFrame(rows).sort_values(["subject", "trials_per_class", "mode"])
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.csv, index=False)
    print(f"Saved results to: {args.csv.resolve()}")


if __name__ == "__main__":
    main()
