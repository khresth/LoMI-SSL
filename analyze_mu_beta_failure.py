from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

from eval_mu_beta_ssl import EncoderNet


def load_bnci_subject_embeddings(weights: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)
    x, y, m = paradigm.get_data(dataset=dataset, subjects=[1])
    train_mask = (m["session"] == "0train").to_numpy()
    x_train, y_train = x[train_mask].astype(np.float32), y[train_mask]
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    model = EncoderNet(n_chans=x_train.shape[1], emb_dim=64)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(x_train)).numpy()
    return z, y_train


def main() -> None:
    csp = pd.read_csv("results_csp_fewshot.csv")
    mu = pd.read_csv("results_mu_beta_ssl_fewshot.csv")
    mu_frozen = mu[mu["mode"] == "mu_beta_ssl_frozen_linear_probe"]
    mu_ft = mu[mu["mode"] == "mu_beta_ssl_full_finetune"]

    z, y = load_bnci_subject_embeddings(Path("mu_beta_ssl_encoder_bnci2014_001.pt"))
    y_enc = LabelEncoder().fit_transform(y)
    z2 = TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto").fit_transform(z)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    agg_csp = csp.groupby("trials_per_class")["mean_accuracy"].mean().reset_index()
    agg_fr = mu_frozen.groupby("trials_per_class")["mean_accuracy"].mean().reset_index()
    agg_ft = mu_ft.groupby("trials_per_class")["mean_accuracy"].mean().reset_index()
    axes[0].plot(agg_csp["trials_per_class"], agg_csp["mean_accuracy"], marker="o", label="CSP+LDA")
    axes[0].plot(agg_fr["trials_per_class"], agg_fr["mean_accuracy"], marker="o", label="mu/beta frozen")
    axes[0].plot(agg_ft["trials_per_class"], agg_ft["mean_accuracy"], marker="o", label="mu/beta finetune")
    axes[0].set_title("Ablation hint: frozen underperforms")
    axes[0].set_xlabel("Trials per class")
    axes[0].set_ylabel("Mean accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    scatter = axes[1].scatter(z2[:, 0], z2[:, 1], c=y_enc, cmap="coolwarm", s=18, alpha=0.8)
    axes[1].set_title("t-SNE of mu/beta embeddings (BNCI S1 train)")
    axes[1].set_xlabel("t-SNE 1")
    axes[1].set_ylabel("t-SNE 2")
    axes[1].grid(alpha=0.3)
    fig.colorbar(scatter, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("mu_beta_failure_analysis.png", dpi=150)
    print("Saved mu_beta_failure_analysis.png")


if __name__ == "__main__":
    main()
