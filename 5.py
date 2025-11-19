import numpy as np
import matplotlib.pyplot as plt

# Practical 5: Implement Joint Probability Mass Function

def joint_pmf_dice():
    die_outcomes = [1, 2, 3, 4, 5, 6]
    sample_space = [(x, y) for x in die_outcomes for y in die_outcomes]
    joint_pmf = {(x, y): 1/36 for (x, y) in sample_space}

    pmf_matrix = np.zeros((6, 6))
    for x in die_outcomes:
        for y in die_outcomes:
            pmf_matrix[x-1, y-1] = joint_pmf[(x, y)]

    plt.figure(figsize=(7, 6))
    plt.imshow(pmf_matrix, cmap="Blues", interpolation="nearest")
    for i in range(6):
        for j in range(6):
            plt.text(j, i, f"{pmf_matrix[i, j]:.3f}", ha="center", va="center", color="black")

    plt.title("Joint PMF of Two 6-Sided Dice")
    plt.xlabel("Outcome of Dice 2")
    plt.ylabel("Outcome of Dice 1")
    plt.colorbar(label="Probability")
    plt.xticks(np.arange(6), die_outcomes)
    plt.yticks(np.arange(6), die_outcomes)
    plt.show()

if __name__ == '__main__':
    joint_pmf_dice()
