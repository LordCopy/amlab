import numpy as np
from scipy.stats import norm

# Sample data
data = [52, 50, 53, 49, 48, 51, 54, 50, 52, 49]

# Hypothesized mean
mu0 = 50

# Step 1: Convert to array
data = np.array(data)

# Step 2: Compute sample mean, std dev, and n
x_bar = np.mean(data)
sigma = np.std(data, ddof=1)
n = len(data)

# Step 3: Compute Z-statistic
z = (x_bar - mu0) / (sigma / np.sqrt(n))

# Step 4: Compute p-value (two-tailed)
p_value = 2 * (1 - norm.cdf(abs(z)))

# Step 5: Print results
print("Sample Mean:", x_bar)
print("Z-statistic:", z)
print("P-value:", p_value)

if p_value < 0.05:
    print("Conclusion: Reject H0")
else:
    print("Conclusion: Fail to Reject H0")
