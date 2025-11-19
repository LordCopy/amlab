import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Practical 6 — Central Limit Theorem demonstration

def clt_demo(sample_size=100000, num_samples=1000):
    sample_means = []
    sample_std_devs = []
    for _ in range(num_samples):
        s = np.random.uniform(0,1,sample_size)
        sample_means.append(np.mean(s))
        sample_std_devs.append(np.std(s))

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    kde_means = stats.gaussian_kde(sample_means)
    x_means = np.linspace(min(sample_means), max(sample_means), 100)
    plt.plot(x_means, kde_means(x_means), color='g')
    plt.title('Distribution of Sample Means')

    plt.subplot(1,2,2)
    kde_sds = stats.gaussian_kde(sample_std_devs)
    x_sds = np.linspace(min(sample_std_devs), max(sample_std_devs), 100)
    plt.plot(x_sds, kde_sds(x_sds), color='b')
    plt.title('Distribution of Sample Std Devs')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    clt_demo()
