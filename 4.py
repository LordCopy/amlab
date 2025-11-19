import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Practical 4: Discrete and Continuous Random Variables (Binomial and Normal Distributions)

def plot_distributions():
    # 1. BINOMIAL DISTRIBUTION
    n = 10
    p = 0.25 * 2
    x_bino = np.arange(0, n + 1)
    pmf_bino = stats.binom.pmf(x_bino, n, p)

    # 2. NORMAL DISTRIBUTION
    mu = 0
    sigma = 1
    x_norm = np.linspace(-4, 4, 100)
    pmf_norm = stats.norm.pdf(x_norm, mu, sigma)

    # 3. PLOTTING
    plt.figure(figsize=(12, 5))

    # Binomial Distribution
    plt.subplot(1, 2, 1)
    plt.stem(x_bino, pmf_bino, basefmt=" ")
    plt.title("Binomial Distribution (n=10, p=0.5)")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.grid(True)

    # Normal Distribution
    plt.subplot(1, 2, 2)
    plt.plot(x_norm, pmf_norm, color="red")
    plt.title("Normal Distribution (μ=0, σ=1)")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_distributions()
