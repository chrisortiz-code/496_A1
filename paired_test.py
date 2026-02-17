import numpy as np
from scipy import stats


def paired_t_test(a, b, alpha=0.10):
    """
    Performs paired t-test between two configurations.

    Parameters:
        a, b : lists or numpy arrays of shape (n,)
               accuracies for each seed
        alpha : significance level (default 0.10 -> 90% CI)

    Returns:
        dict with statistics
    """
    a = np.array(a)
    b = np.array(b)

    assert len(a) == len(b), "Paired test requires equal length samples."

    n = len(a)
    df = n - 1

    # Differences
    d = a - b
    mean_d = np.mean(d)
    sd_d = np.std(d, ddof=1)
    se_d = sd_d / np.sqrt(n)

    # t-statistic
    t_stat = mean_d / se_d

    # two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # critical value for CI
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    ci_lower = mean_d - t_crit * se_d
    ci_upper = mean_d + t_crit * se_d

    # Cohen's d for paired samples
    cohen_d = mean_d / sd_d

    return {
        "mean_diff": mean_d,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci": (ci_lower, ci_upper),
        "cohen_d": cohen_d,
        "df": df
    }


# =========================
# Example usage
# =========================
if __name__ == "__main__":

    # Replace with your actual 5-seed accuracies
    cotrain = [0.878, 0.879, 0.876, 0.878, 0.877]
    labelprop = [0.864, 0.862, 0.865, 0.863, 0.864]

    results = paired_t_test(cotrain, labelprop)

    print("\nPaired t-test Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
