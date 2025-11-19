import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Practical 6: Implement Central Limit Theorem and Plot Probability Distribution

def clt_demo(sample_size=100000, num_samples=1000):
    sample_means = []
    sample_std_devs = []

    for _ in range(num_samples):
        sample = np.random.uniform(0, 1, sample_size)
        sample_means.append(np.mean(sample))
        sample_std_devs.append(np.std(sample))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    kde_means = stats.gaussian_kde(sample_means)
    x_means = np.linspace(min(sample_means), max(sample_means), 100)
    plt.plot(x_means, kde_means(x_means), color='g')
    plt.title('Distribution of Sample Means (KDE)')
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    kde_std_devs = stats.gaussian_kde(sample_std_devs)
    x_std_devs = np.linspace(min(sample_std_devs), max(sample_std_devs), 100)
    plt.plot(x_std_devs, kde_std_devs(x_std_devs), color='b')
    plt.title('Distribution of Sample Standard Deviations (KDE)')
    plt.xlabel('Sample Standard Deviation')
    plt.ylabel('Density')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    clt_demo()
