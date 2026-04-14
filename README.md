# FewShotMI-Bench

FewShotMI-Bench is a standardized benchmark package for few-shot motor imagery (MI) calibration.

The benchmark aims to evaluate calibration performance under strict low-label settings rather than optimistic within-session cross-validation alone. The package provides leakage-aware split logic, fixed trial budgets, repeatable sampling, multi-baseline comparisons, and deployment-oriented measurements for practical BCI workflows.

## Abstract-Style Summary

FewShotMI-Bench targets the clinical and real-world MI calibration problem: adapting to a new user with minimal labeled data. The benchmark protocol uses fixed few-shot budgets per class with repeated sampling and predefined train/test partitions to reduce leakage risk and improve reproducibility. Baselines include classical CSP+LDA, deep EEGNet from scratch, and compact SSL variants (masked reconstruction and mu/beta-contrastive). Cross-dataset transfer is included to expose generalization gaps between source and target datasets. The package also includes deployment metrics such as ONNX CPU latency, memory footprint checks, and int8 accuracy drop analysis.

The benchmark aims to support rigorous and reproducible SSL/transfer research for practical MI-BCI calibration.

## Repository Layout

- `eval/`: main evaluation scripts (`eval_*.py`)
- `data/`: result CSV outputs (`results_*.csv`)
- `images/paper/`: publication-ready figure set
- `paper/`: manuscript sources (including `fewshotmi_bench.tex`)

## Current Known Gaps

- EEGNet few-shot result CSVs are still missing for complete multi-method tables on all datasets.
- BNCI2014-004 masked SSL CSV is not yet generated.
- Physionet run-based split only supports feasible trial budgets (5/10), so 20/50 are not available there under the current split.
