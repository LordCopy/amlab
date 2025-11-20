import numpy as np
import matplotlib.pyplot as plt

# 1. Define outcomes
die_outcomes = [1, 2, 3, 4, 5, 6]

# 2. Create sample space (all pairs)
sample_space = [(x, y) for x in die_outcomes for y in die_outcomes]

# 3. Joint PMF dictionary
joint_pmf = {}
for (x, y) in sample_space:
    joint_pmf[(x, y)] = 1 / 36     # equal probability

# 4. Convert PMF to 6x6 matrix
pmf_matrix = np.zeros((6, 6))
for x in die_outcomes:
    for y in die_outcomes:
        pmf_matrix[x-1, y-1] = joint_pmf[(x, y)]

# 5. Heatmap Plot
plt.figure(figsize=(7, 6))
plt.imshow(pmf_matrix, cmap="Blues", interpolation="nearest")

# show probability value on each cell
for i in range(6):
    for j in range(6):
        plt.text(j, i, f"{pmf_matrix[i, j]:.3f}",
                 ha="center", va="center", color="black")

plt.title("Joint PMF of Two Dice")
plt.xlabel("Outcome of Dice 2")
plt.ylabel("Outcome of Dice 1")
plt.xticks(np.arange(6), die_outcomes)
plt.yticks(np.arange(6), die_outcomes)
plt.colorbar(label="Probability")
plt.show()
