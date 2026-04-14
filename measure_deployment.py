from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from moabb.datasets import BNCI2014_001
from moabb.paradigms import LeftRightImagery
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from eval_mu_beta_ssl import EncoderNet


def main() -> None:
    dataset = BNCI2014_001()
    paradigm = LeftRightImagery(fmin=8, fmax=32, resample=128)
    x, y, m = paradigm.get_data(dataset=dataset, subjects=[1])
    tr = (m["session"] == "0train").to_numpy()
    te = (m["session"] == "1test").to_numpy()
    x_tr, y_tr = x[tr].astype(np.float32), y[tr]
    x_te, y_te = x[te].astype(np.float32), y[te]
    mean = x_tr.mean(axis=(0, 2), keepdims=True)
    std = x_tr.std(axis=(0, 2), keepdims=True) + 1e-6
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std

    model = EncoderNet(n_chans=x_tr.shape[1], emb_dim=64)
    model.load_state_dict(torch.load("mu_beta_ssl_encoder_bnci2014_001.pt", map_location="cpu"))
    model.eval()

    dummy = torch.from_numpy(x_te[:1])
    torch.onnx.export(
        model,
        dummy,
        "mu_beta_encoder.onnx",
        input_names=["eeg"],
        output_names=["embedding"],
        dynamic_axes={"eeg": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    sess = ort.InferenceSession("mu_beta_encoder.onnx", providers=["CPUExecutionProvider"])
    for _ in range(20):
        _ = sess.run(None, {"eeg": x_te[:1]})
    t0 = time.perf_counter()
    n_runs = 200
    for _ in range(n_runs):
        _ = sess.run(None, {"eeg": x_te[:1]})
    latency_ms = (time.perf_counter() - t0) * 1000 / n_runs

    with torch.no_grad():
        z_tr = model(torch.from_numpy(x_tr)).numpy()
        z_te = model(torch.from_numpy(x_te)).numpy()
    le = LabelEncoder()
    y_tr_e = le.fit_transform(y_tr)
    y_te_e = le.transform(y_te)
    clf = LogisticRegression(max_iter=500).fit(z_tr, y_tr_e)
    fp_acc = accuracy_score(y_te_e, clf.predict(z_te))

    head = nn.Linear(z_tr.shape[1], len(le.classes_))
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(clf.coef_.astype(np.float32)))
        head.bias.copy_(torch.from_numpy(clf.intercept_.astype(np.float32)))
    q_head = torch.ao.quantization.quantize_dynamic(head, {nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        logits_q = q_head(torch.from_numpy(z_te.astype(np.float32))).numpy()
    int8_acc = accuracy_score(y_te_e, np.argmax(logits_q, axis=1))
    int8_drop = fp_acc - int8_acc

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        x_small = torch.from_numpy(x_tr[:64]).to(device)
        y_small = torch.from_numpy(y_tr_e[:64]).to(device)
        head_cuda = nn.Linear(64, len(le.classes_)).to(device)
        opt = torch.optim.Adam(list(model.parameters()) + list(head_cuda.parameters()), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(20):
            opt.zero_grad()
            logits = head_cuda(model(x_small))
            loss = loss_fn(logits, y_small)
            loss.backward()
            opt.step()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    with open("deployment_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Deployment metrics\n")
        f.write("==================\n")
        f.write(f"ONNX CPU latency (batch=1): {latency_ms:.3f} ms\n")
        f.write(f"Linear probe float32 accuracy: {fp_acc:.4f}\n")
        f.write(f"Linear probe int8 accuracy: {int8_acc:.4f}\n")
        f.write(f"Int8 accuracy drop: {int8_drop:.4f}\n")
        if torch.cuda.is_available():
            f.write(f"Peak VRAM during few-shot FT mini-run: {peak_vram_mb:.2f} MB\n")
        else:
            f.write("Peak VRAM during few-shot FT mini-run: N/A (CUDA unavailable)\n")
    print("Saved deployment_metrics.txt")


if __name__ == "__main__":
    main()
