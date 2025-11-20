import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. BINOMIAL DISTRIBUTION
n = 10               # Number of trials
p = 0.25 * 2         # Probability of success (0.5)

x_bino = np.arange(0, n + 1)
pmf_bino = stats.binom.pmf(x_bino, n, p)

# 2. NORMAL DISTRIBUTION
mu = 0               # Mean
sigma = 1            # Standard deviation

x_norm = np.linspace(-4, 4, 100)
pdf_norm = stats.norm.pdf(x_norm, mu, sigma)

# 3. PLOTTING
plt.figure(figsize=(12, 5))

# Binomial Plot
plt.subplot(1, 2, 1)
plt.stem(x_bino, pmf_bino, basefmt=" ")
plt.title("Binomial Distribution (n=10, p=0.5)")
plt.xlabel("Number of Successes")
plt.ylabel("Probability")
plt.grid(True)

# Normal Plot
plt.subplot(1, 2, 2)
plt.plot(x_norm, pdf_norm)
plt.title("Normal Distribution (μ=0, σ=1)")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.grid(True)

plt.show()
