# Central Limit Theorem Demo
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

sample_size = 100000
num_samples = 100000
sample_means = []

for i in range(num_samples):
    sample = np.random.uniform(0, 1, sample_size)
    sample_means.append(np.mean(sample))

# KDE for sample means
kde = gaussian_kde(sample_means)
x_vals = np.linspace(min(sample_means), max(sample_means), 100)

plt.plot(x_vals, kde(x_vals))
plt.title("Central Limit Theorem – Distribution of Sample Means")
plt.xlabel("Sample Mean")
plt.ylabel("Density")
plt.grid(True)
plt.show()
