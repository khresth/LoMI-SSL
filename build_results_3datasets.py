from pathlib import Path

import pandas as pd


def load_optional(path: Path, dataset: str, experiment: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "dataset",
                "experiment",
                "subject",
                "trials_per_class",
                "mode",
                "mean_accuracy",
                "std_accuracy",
            ]
        )
    df = pd.read_csv(path)
    if "mode" not in df.columns:
        df["mode"] = experiment
    df["dataset"] = dataset
    df["experiment"] = experiment
    wanted = [
        "dataset",
        "experiment",
        "subject",
        "trials_per_class",
        "mode",
        "mean_accuracy",
        "std_accuracy",
    ]
    for col in wanted:
        if col not in df.columns:
            df[col] = pd.NA
    return df[wanted]


def main() -> None:
    rows = []
    rows.append(
        load_optional(
            Path("results_csp_fewshot.csv"),
            dataset="BNCI2014_001",
            experiment="csp_lda",
        )
    )
    rows.append(
        load_optional(
            Path("results_ssl_fewshot.csv"),
            dataset="BNCI2014_001",
            experiment="masked_ssl",
        )
    )
    rows.append(
        load_optional(
            Path("results_mu_beta_ssl_fewshot.csv"),
            dataset="BNCI2014_001",
            experiment="mu_beta_ssl",
        )
    )
    rows.append(
        load_optional(
            Path("results_mu_beta_physionetmi.csv"),
            dataset="PhysionetMI_1_20",
            experiment="mu_beta_ssl",
        )
    )
    rows.append(
        load_optional(
            Path("results_masked_ssl_physionetmi.csv"),
            dataset="PhysionetMI_1_20",
            experiment="masked_ssl",
        )
    )
    rows.append(
        load_optional(
            Path("results_masked_ssl_bnci2014_004.csv"),
            dataset="BNCI2014_004",
            experiment="masked_ssl",
        )
    )
    rows.append(
        load_optional(
            Path("results_cross_bnci_to_physio.csv"),
            dataset="PhysionetMI_1_20",
            experiment="cross_bnci_to_physio",
        )
    )
    rows.append(
        load_optional(
            Path("results_eegnet_fewshot_physionetmi.csv"),
            dataset="PhysionetMI_1_20",
            experiment="eegnet_scratch",
        )
    )
    rows.append(
        load_optional(
            Path("results_eegnet_physio.csv"),
            dataset="PhysionetMI_1_20",
            experiment="eegnet_scratch",
        )
    )
    rows.append(
        load_optional(
            Path("results_eegnet_fewshot_bnci2014_001.csv"),
            dataset="BNCI2014_001",
            experiment="eegnet_scratch",
        )
    )
    rows.append(
        load_optional(
            Path("results_eegnet_bnci.csv"),
            dataset="BNCI2014_001",
            experiment="eegnet_scratch",
        )
    )
    out = pd.concat(rows, ignore_index=True)
    out.to_csv("results_3datasets.csv", index=False)
    print("Wrote results_3datasets.csv")


if __name__ == "__main__":
    main()
