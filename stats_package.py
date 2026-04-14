import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.power import TTestPower


def cohens_d_paired(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def main() -> None:
    csp = pd.read_csv("results_csp_fewshot.csv")
    ssl = pd.read_csv("results_mu_beta_ssl_fewshot.csv")
    ssl = ssl[ssl["mode"] == "mu_beta_ssl_full_finetune"]

    pivot_csp = csp.pivot(index="subject", columns="trials_per_class", values="mean_accuracy")
    pivot_ssl = ssl.pivot(index="subject", columns="trials_per_class", values="mean_accuracy")
    shots = [5, 10, 20, 50]
    m_tests = len(shots)
    alpha_bonf = 0.05 / m_tests

    rows = []
    power_calc = TTestPower()
    for s in shots:
        x = pivot_ssl[s].dropna().to_numpy()
        y = pivot_csp[s].dropna().to_numpy()
        t_stat, p_val = ttest_rel(x, y)
        d = cohens_d_paired(x, y)
        achieved_power = power_calc.solve_power(effect_size=abs(d), nobs=len(x), alpha=0.05)
        n_for_08 = power_calc.solve_power(effect_size=max(abs(d), 1e-4), power=0.8, alpha=0.05)
        rows.append(
            {
                "shots": s,
                "ssl_mean": float(np.mean(x)),
                "csp_mean": float(np.mean(y)),
                "delta": float(np.mean(x) - np.mean(y)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "p_bonferroni": float(min(1.0, p_val * m_tests)),
                "bonf_significant": bool(p_val < alpha_bonf),
                "cohens_d_paired": d,
                "achieved_power_n9": float(achieved_power),
                "required_n_for_0_8_power": float(n_for_08),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv("stats_summary.csv", index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("Saved stats_summary.csv")


if __name__ == "__main__":
    main()
