from pathlib import Path
import re

import matplotlib.pyplot as plt


def parse_metrics(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    metrics = {}
    patterns = {
        "latency_ms": r"ONNX CPU latency \(batch=1\):\s*([0-9.]+)",
        "int8_drop": r"Int8 accuracy drop:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        metrics[key] = float(m.group(1)) if m else float("nan")
    vram_match = re.search(r"Peak VRAM.*:\s*([0-9.]+)\s*MB", text)
    metrics["peak_vram_mb"] = float(vram_match.group(1)) if vram_match else 0.0
    return metrics


def main() -> None:
    m = parse_metrics(Path("deployment_metrics.txt"))
    names = ["CPU latency (ms)", "Peak VRAM (MB)", "Int8 acc drop"]
    values = [m["latency_ms"], m["peak_vram_mb"], m["int8_drop"]]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, values)
    plt.title("Deployment metrics")
    plt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    out = Path("deployment_metrics.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out.resolve()}")


if __name__ == "__main__":
    main()
