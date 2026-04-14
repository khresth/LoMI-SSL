from pathlib import Path
import argparse
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from moabb import set_download_dir
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from scipy.signal import butter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


TRIALS_PER_CLASS = (5, 10, 20, 50)
N_REPEATS = 5


class MuBetaContrastiveDataset(Dataset):
    def __init__(
        self,
        x_full: np.ndarray,
        x_mu_beta: np.ndarray,
        x_delta: np.ndarray,
        x_theta: np.ndarray,
        x_gamma: np.ndarray,
        seed: int,
    ) -> None:
        self.x_full = torch.from_numpy(x_full.astype(np.float32, copy=False))
        self.x_mu_beta = torch.from_numpy(x_mu_beta.astype(np.float32, copy=False))
        self.non_mi_bands = [
            torch.from_numpy(x_delta.astype(np.float32, copy=False)),
            torch.from_numpy(x_theta.astype(np.float32, copy=False)),
            torch.from_numpy(x_gamma.astype(np.float32, copy=False)),
        ]
        self.seed = seed

    def __len__(self) -> int:
        return self.x_full.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx)
        neg_band = int(rng.integers(0, len(self.non_mi_bands)))
        x_non_mi = self.non_mi_bands[neg_band][idx]
        return self.x_full[idx], self.x_mu_beta[idx], x_non_mi


class EncoderNet(nn.Module):
    def __init__(self, n_chans: int, emb_dim: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_chans, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, emb_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        emb = self.pool(feats).squeeze(-1)
        return emb


class MuBetaContrastiveModel(nn.Module):
    def __init__(self, n_chans: int, emb_dim: int = 64, proj_dim: int = 32) -> None:
        super().__init__()
        self.encoder = EncoderNet(n_chans=n_chans, emb_dim=emb_dim)
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ELU(),
            nn.Linear(emb_dim, proj_dim),
        )

    def encode_project(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        z = self.projector(z)
        return F.normalize(z, p=2, dim=1)


class ClassifierNet(nn.Module):
    def __init__(self, encoder: EncoderNet, emb_dim: int, n_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.head(z)


@dataclass
class SubjectData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bandpass_filter(x: np.ndarray, sfreq: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    sos = butter(order, [fmin / nyq, fmax / nyq], btype="bandpass", output="sos")
    # filter over time axis
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32, copy=False)


def nt_xent_mu_beta_loss(
    z_full: torch.Tensor,
    z_mu_beta: torch.Tensor,
    z_non_mi: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    sim_pos = torch.sum(z_full * z_mu_beta, dim=1) / temperature  # [B]
    sim_mu = torch.matmul(z_full, z_mu_beta.T) / temperature  # [B,B]
    sim_non = torch.matmul(z_full, z_non_mi.T) / temperature  # [B,B]

    exp_pos = torch.exp(sim_pos)
    denom = torch.exp(sim_mu).sum(dim=1) + torch.exp(sim_non).sum(dim=1)
    loss = -torch.log(exp_pos / (denom + 1e-12) + 1e-12)
    return loss.mean()


def load_subject_data(subject: int, dataset: BNCI2014_001, paradigm: LeftRightImagery) -> SubjectData:
    x, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject])
    train_mask = metadata["session"] == "0train"
    test_mask = metadata["session"] == "1test"
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(
            f"Subject {subject} does not contain expected 0train/1test sessions."
        )
    return SubjectData(
        x_train=x[train_mask.to_numpy()],
        y_train=y[train_mask.to_numpy()],
        x_test=x[test_mask.to_numpy()],
        y_test=y[test_mask.to_numpy()],
    )


def fit_standardizer(x_all_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_all_train.mean(axis=(0, 2), keepdims=True)
    std = x_all_train.std(axis=(0, 2), keepdims=True) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32, copy=False)


def pretrain_mu_beta_ssl(
    x_full: np.ndarray,
    sfreq: float,
    n_chans: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    temperature: float,
    seed: int,
) -> MuBetaContrastiveModel:
    set_seed(seed)

    x_mu_beta = bandpass_filter(x_full, sfreq=sfreq, fmin=8.0, fmax=30.0)
    x_delta = bandpass_filter(x_full, sfreq=sfreq, fmin=1.0, fmax=4.0)
    x_theta = bandpass_filter(x_full, sfreq=sfreq, fmin=4.0, fmax=8.0)
    x_gamma = bandpass_filter(x_full, sfreq=sfreq, fmin=30.0, fmax=45.0)

    ds = MuBetaContrastiveDataset(
        x_full=x_full,
        x_mu_beta=x_mu_beta,
        x_delta=x_delta,
        x_theta=x_theta,
        x_gamma=x_gamma,
        seed=seed,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = MuBetaContrastiveModel(n_chans=n_chans, emb_dim=64, proj_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for xb_full, xb_mu, xb_non in loader:
            xb_full = xb_full.to(device)
            xb_mu = xb_mu.to(device)
            xb_non = xb_non.to(device)
            optimizer.zero_grad()
            z_full = model.encode_project(xb_full)
            z_mu = model.encode_project(xb_mu)
            z_non = model.encode_project(xb_non)
            loss = nt_xent_mu_beta_loss(z_full, z_mu, z_non, temperature=temperature)
            loss.backward()
            optimizer.step()
    return model


def sample_fewshot_indices(y_train: np.ndarray, trials_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train)
    sampled_parts = []
    for cls in classes:
        cls_idx = np.flatnonzero(y_train == cls)
        sampled = rng.choice(cls_idx, size=trials_per_class, replace=False)
        sampled_parts.append(sampled)
    sampled_idx = np.concatenate(sampled_parts)
    rng.shuffle(sampled_idx)
    return sampled_idx


def evaluate_frozen_linear_probe(
    encoder: EncoderNet,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> float:
    encoder.eval()
    with torch.no_grad():
        z_cal = encoder(torch.from_numpy(x_cal).to(device))
        z_test = encoder(torch.from_numpy(x_test).to(device))
    z_cal_np = z_cal.cpu().numpy()
    z_test_np = z_test.cpu().numpy()

    le = LabelEncoder()
    y_cal_enc = le.fit_transform(y_cal)
    y_test_enc = le.transform(y_test)
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(z_cal_np, y_cal_enc)
    y_pred = clf.predict(z_test_np)
    return accuracy_score(y_test_enc, y_pred)


def evaluate_full_finetune(
    encoder: EncoderNet,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    finetune_epochs: int,
    finetune_lr: float,
) -> float:
    le = LabelEncoder()
    y_cal_enc = le.fit_transform(y_cal)
    y_test_enc = le.transform(y_test)

    model = ClassifierNet(encoder=encoder, emb_dim=64, n_classes=len(le.classes_)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
    loss_fn = nn.CrossEntropyLoss()

    x_cal_t = torch.from_numpy(x_cal).to(device)
    y_cal_t = torch.from_numpy(y_cal_enc.astype(np.int64)).to(device)
    x_test_t = torch.from_numpy(x_test).to(device)

    model.train()
    for _ in range(finetune_epochs):
        optimizer.zero_grad()
        logits = model(x_cal_t)
        loss = loss_fn(logits, y_cal_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits_test = model(x_test_t)
        y_pred = torch.argmax(logits_test, dim=1).cpu().numpy()
    return accuracy_score(y_test_enc, y_pred)


def evaluate_subject(
    subject: int,
    subject_data: SubjectData,
    pretrained_encoder_state: dict[str, torch.Tensor],
    device: torch.device,
    finetune_epochs: int,
    finetune_lr: float,
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for trials_per_class in TRIALS_PER_CLASS:
        frozen_scores: list[float] = []
        finetune_scores: list[float] = []
        for repeat in range(N_REPEATS):
            sample_seed = 1000 * subject + 100 * trials_per_class + repeat
            sampled_idx = sample_fewshot_indices(
                subject_data.y_train, trials_per_class=trials_per_class, seed=sample_seed
            )
            x_cal = subject_data.x_train[sampled_idx]
            y_cal = subject_data.y_train[sampled_idx]

            enc_frozen = EncoderNet(n_chans=subject_data.x_train.shape[1], emb_dim=64).to(device)
            enc_frozen.load_state_dict(pretrained_encoder_state)
            frozen_acc = evaluate_frozen_linear_probe(
                encoder=enc_frozen,
                x_cal=x_cal,
                y_cal=y_cal,
                x_test=subject_data.x_test,
                y_test=subject_data.y_test,
                device=device,
            )
            frozen_scores.append(frozen_acc)

            enc_finetune = EncoderNet(n_chans=subject_data.x_train.shape[1], emb_dim=64).to(device)
            enc_finetune.load_state_dict(pretrained_encoder_state)
            finetune_acc = evaluate_full_finetune(
                encoder=enc_finetune,
                x_cal=x_cal,
                y_cal=y_cal,
                x_test=subject_data.x_test,
                y_test=subject_data.y_test,
                device=device,
                finetune_epochs=finetune_epochs,
                finetune_lr=finetune_lr,
            )
            finetune_scores.append(finetune_acc)

        rows.append(
            {
                "subject": subject,
                "trials_per_class": trials_per_class,
                "mode": "mu_beta_ssl_frozen_linear_probe",
                "mean_accuracy": float(np.mean(frozen_scores)),
                "std_accuracy": float(np.std(frozen_scores, ddof=0)),
            }
        )
        rows.append(
            {
                "subject": subject,
                "trials_per_class": trials_per_class,
                "mode": "mu_beta_ssl_full_finetune",
                "mean_accuracy": float(np.mean(finetune_scores)),
                "std_accuracy": float(np.std(finetune_scores, ddof=0)),
            }
        )
    return rows


def plot_comparison(results: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    agg = (
        results.groupby(["mode", "trials_per_class"])["mean_accuracy"]
        .mean()
        .reset_index()
        .sort_values("trials_per_class")
    )
    for mode in ["csp_lda", "mu_beta_ssl_frozen_linear_probe", "mu_beta_ssl_full_finetune"]:
        mode_df = agg[agg["mode"] == mode]
        if mode_df.empty:
            continue
        plt.plot(
            mode_df["trials_per_class"],
            mode_df["mean_accuracy"],
            marker="o",
            linewidth=2,
            label=mode,
        )
    plt.xlabel("Trials per class")
    plt.ylabel("Mean accuracy across subjects")
    plt.title("Few-shot calibration: CSP+LDA vs mu/beta SSL")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="mu/beta contrastive SSL pretraining + few-shot calibration on BNCI2014_001."
    )
    parser.add_argument("--csv", type=Path, default=Path("results_mu_beta_ssl_fewshot.csv"))
    parser.add_argument("--plot", type=Path, default=Path("results_mu_beta_ssl_vs_csp.png"))
    parser.add_argument("--weights", type=Path, default=Path("mu_beta_ssl_encoder_bnci2014_001.pt"))
    parser.add_argument("--csp-csv", type=Path, default=Path("results_csp_fewshot.csv"))
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

    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)
    sfreq = 128.0

    print(f"Using device: {device}")
    print(f"Using cache directory: {cache_dir.resolve()}")
    print("Loading all subjects with session split 0train -> 1test...")

    subject_data: dict[int, SubjectData] = {}
    x_pretrain_parts: list[np.ndarray] = []
    for subject in dataset.subject_list:
        sdata = load_subject_data(subject=subject, dataset=dataset, paradigm=paradigm)
        subject_data[subject] = sdata
        x_pretrain_parts.append(sdata.x_train)

    x_pretrain_raw = np.concatenate(x_pretrain_parts, axis=0)
    mean, std = fit_standardizer(x_pretrain_raw)
    x_pretrain = apply_standardizer(x_pretrain_raw, mean, std)
    for subject in dataset.subject_list:
        s = subject_data[subject]
        subject_data[subject] = SubjectData(
            x_train=apply_standardizer(s.x_train, mean, std),
            y_train=s.y_train,
            x_test=apply_standardizer(s.x_test, mean, std),
            y_test=s.y_test,
        )

    print("Pretraining compact mu/beta-contrastive encoder on unlabeled 0train data...")
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
    for subject in dataset.subject_list:
        subject_rows = evaluate_subject(
            subject=subject,
            subject_data=subject_data[subject],
            pretrained_encoder_state=pretrained_encoder_state,
            device=device,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
        )
        rows.extend(subject_rows)
        summary_parts = []
        for tpc in TRIALS_PER_CLASS:
            sub_t = [r for r in subject_rows if r["trials_per_class"] == tpc]
            fz = [r for r in sub_t if r["mode"] == "mu_beta_ssl_frozen_linear_probe"][0]
            ft = [r for r in sub_t if r["mode"] == "mu_beta_ssl_full_finetune"][0]
            summary_parts.append(
                f"{tpc}tpc frozen={float(fz['mean_accuracy']):.4f}+/-{float(fz['std_accuracy']):.4f} "
                f"finetune={float(ft['mean_accuracy']):.4f}+/-{float(ft['std_accuracy']):.4f}"
            )
        print(f"Subject {subject:>2}: " + " | ".join(summary_parts))

    ssl_df = pd.DataFrame(rows).sort_values(["subject", "trials_per_class", "mode"])

    csp_df = pd.read_csv(args.csp_csv)
    csp_df["mode"] = "csp_lda"
    csp_df = csp_df[["subject", "trials_per_class", "mode", "mean_accuracy", "std_accuracy"]]

    combined = pd.concat([csp_df, ssl_df], ignore_index=True)
    combined = combined.sort_values(["mode", "subject", "trials_per_class"]).reset_index(drop=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.csv, index=False)
    print(f"Saved comparison CSV to: {args.csv.resolve()}")

    agg = (
        combined.groupby(["mode", "trials_per_class"])["mean_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_accuracy", "std": "std_across_subjects"})
    )
    print("-" * 84)
    print("Across-subject summary")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    plot_comparison(combined, args.plot)
    print(f"Saved comparison plot to: {args.plot.resolve()}")


if __name__ == "__main__":
    main()
