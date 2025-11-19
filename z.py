import numpy as np
from scipy.stats import norm

# Practical 8 — Z-test for single mean

def z_test_single_mean(sample, mu0, sigma=None, alpha=0.05, alternative="two-sided"):
    sample = np.array(sample)
    n = len(sample)
    x_bar = np.mean(sample)
    if sigma is None:
        sigma = np.std(sample, ddof=1)
    z_stat = (x_bar - mu0) / (sigma / np.sqrt(n))
    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - norm.cdf(z_stat)
    elif alternative == "less":
        p_value = norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    return {"Z-stat": z_stat, "P-value": p_value}

if __name__ == '__main__':
    data = [52,50,53,49,48,51,54,50,52,49]
    print(z_test_single_mean(data, 50))
