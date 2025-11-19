import numpy as np
import matplotlib.pyplot as plt

# Practical 9: Calculate n-Step Transition Probability

def validate_transition_matrix(P):
    P = np.array(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    if np.any(P < 0):
        raise ValueError("Transition matrix cannot have negative entries.")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows of P must sum to 1")
    return P

def n_step_transition_probability(P, n):
    P = validate_transition_matrix(P)
    return np.linalg.matrix_power(P, n)

if __name__ == '__main__':
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.4, 0.2, 0.4]
    ])
    steps = [1, 2, 5, 10, 20]
    for n in steps:
        Pn = n_step_transition_probability(P, n)
        print(f"\n{n}-step transition probability matrix:\n{Pn}")

    state_names = ["S0", "S1", "S2"]
    max_n = 20
    evolution = [n_step_transition_probability(P, n)[0] for n in range(1, max_n + 1)]
    evolution = np.array(evolution)

    plt.figure(figsize=(8, 5))
    for i in range(P.shape[0]):
        plt.plot(range(1, max_n + 1), evolution[:, i], marker='o', label=f"P(S0→S{i})")
    plt.xlabel("Number of steps (n)")
    plt.ylabel("Probability")
    plt.title("n-step Transition Probabilities from S0 to Each State")
    plt.xticks(range(1, max_n + 1))
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
