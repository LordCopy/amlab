# Assignment 4: Demonstrate Discrete & Continuous Random Variables
# Using Real-world Examples (Coin Toss & Height Distribution)

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1️⃣ BINOMIAL DISTRIBUTION (Discrete)
# Example: 10 coin tosses, p=0.5 (probability of heads)

n = 10        # number of trials
p = 0.5       # probability of getting a head
x_bino = np.arange(0, n + 1)

# PMF - Probability Mass Function
pmf_bino = stats.binom.pmf(x_bino, n, p)

# 2️⃣ NORMAL DISTRIBUTION (Continuous)
# Example: Students’ height distribution (mean=170cm, std=10cm)

mu = 170      # mean height
sigma = 10    # standard deviation
x_norm = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
pdf_norm = stats.norm.pdf(x_norm, mu, sigma)

# 3️⃣ PLOTTING BOTH DISTRIBUTIONS
plt.figure(figsize=(12, 5))

# Binomial Distribution Plot
plt.subplot(1, 2, 1)
plt.stem(x_bino, pmf_bino, basefmt=" ")
plt.title("Binomial Distribution (10 Coin Tosses)")
plt.xlabel("Number of Heads")
plt.ylabel("Probability")
plt.grid(True)

# Normal Distribution Plot
plt.subplot(1, 2, 2)
plt.plot(x_norm, pdf_norm, color="red")
plt.title("Normal Distribution of Student Heights")
plt.xlabel("Height (cm)")
plt.ylabel("Probability Density")
plt.grid(True)

plt.tight_layout()
plt.show()
