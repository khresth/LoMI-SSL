import os
from pathlib import Path

from mne.decoding import CSP
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def main() -> None:
    # Keep all downloaded files in a local cache directory.
    cache_dir = Path.cwd() / "moabb_data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MNE_DATA"] = str(cache_dir.resolve())

    print(f"Using cache directory: {cache_dir.resolve()}")
    print("Loading BNCI2014001 (BCI IV 2a)...")

    dataset = BNCI2014001()
    subject = dataset.subject_list[0]

    # Minimal MI preprocessing: binary MI classes, band-pass through paradigm, and resample.
    paradigm = MotorImagery(n_classes=2, fmin=8, fmax=32, resample=128)
    X, y, _ = paradigm.get_data(dataset=dataset, subjects=[subject])

    print(f"Subject: {subject}")
    print(f"Epochs shape: {X.shape}")
    print(f"Class distribution: {dict(zip(*__import__('numpy').unique(y, return_counts=True)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = make_pipeline(
        CSP(n_components=8, log=True, norm_trace=False),
        LinearDiscriminantAnalysis(),
    )

    print("Training CSP + LDA baseline...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Baseline accuracy (subject {subject}, single split): {acc:.4f}")


if __name__ == "__main__":
    main()
