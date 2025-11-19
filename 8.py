import numpy as np
from scipy.stats import norm

# Practical 8: Perform Large Sample Test for Single Mean (Z-test)

def z_test_single_mean(sample, mu0, sigma=None, alpha=0.05, alternative="two-sided"):
    sample = np.array(sample)
    n = len(sample)
    x_bar = np.mean(sample)
    
    if sigma is None:
        sigma = np.std(sample, ddof=1)
    
    # Compute Z-statistic
    z_stat = (x_bar - mu0) / (sigma / np.sqrt(n))
    
    # Compute p-value based on test type
    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    elif alternative == "greater":
        p_value = 1 - norm.cdf(z_stat)
    elif alternative == "less":
        p_value = norm.cdf(z_stat)
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    
    reject = p_value <= alpha
    
    return {
        "Sample Mean": x_bar,
        "Hypothesized Mean": mu0,
        "Z-statistic": z_stat,
        "P-value": p_value,
        "Conclusion": "Reject H₀" if reject else "Fail to Reject H₀"
    }

if __name__ == '__main__':
    data = [52, 50, 53, 49, 48, 51, 54, 50, 52, 49]
    mu0 = 50
    alpha = 0.05

    result = z_test_single_mean(data, mu0, alpha=alpha, sigma=None, alternative='two-sided')
    print(result)
