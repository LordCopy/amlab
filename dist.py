import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Practical 4 — Binomial and Normal distributions plotting

def plot_distributions():
    n = 10
    p = 0.25 * 2
    x_bino = np.arange(0, n+1)
    pmf_bino = stats.binom.pmf(x_bino, n, p)

    x_norm = np.linspace(-4, 4, 100)
    pmf_norm = stats.norm.pdf(x_norm, 0, 1)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.stem(x_bino, pmf_bino, basefmt=' ')
    plt.title('Binomial (n=10,p=0.5)')
    plt.xlabel('Number of Successes')

    plt.subplot(1,2,2)
    plt.plot(x_norm, pmf_norm, color='red')
    plt.title('Normal (μ=0,σ=1)')
    plt.xlabel('Value')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_distributions()
