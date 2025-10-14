import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. BINOMIAL DISTRIBUTION

# Number of trials
n = 10
# Probability of success (here 0.25*2 = 0.5)
p = 0.25 * 2

# Possible values for number of successes (0, 1, ..., n)
x_bino = np.arange(0, n + 1)

# Probability Mass Function (PMF) for binomial distribution
pmf_bino = stats.binom.pmf(x_bino, n, p)

# 2. NORMAL DISTRIBUTION

# Mean (μ) and Standard Deviation (σ)
mu = 0
sigma = 1

# Generate x values in range [-4, 4]
x_norm = np.linspace(-4, 4, 100)

# Probability Density Function (PDF) for normal distribution
pmf_norm = stats.norm.pdf(x_norm, mu, sigma)

# 3. PLOTTING

plt.figure(figsize=(12, 5))  # Create figure with 2 plots

# Binomial Distribution (Discrete)
plt.subplot(1, 2, 1)
plt.stem(x_bino, pmf_bino, basefmt=" ")
plt.title("Binomial Distribution (n=10, p=0.5)")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.grid(True)

# Normal Distribution (Continuous)
plt.subplot(1, 2, 2)
plt.plot(x_norm, pmf_norm, color="red")
plt.title("Normal Distribution (μ=0, σ=1)")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.grid(True)

# Adjust layout so plots don't overlap
plt.tight_layout()
plt.show()
