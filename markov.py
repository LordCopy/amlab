import numpy as np
import matplotlib.pyplot as plt

# Practical 9 — n-step transition probabilities for a Markov chain

def validate_transition_matrix(P):
    P = np.array(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be square")
    if np.any(P < 0):
        raise ValueError("Negative entries not allowed")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows must sum to 1")
    return P

def n_step_transition_probability(P, n):
    P = validate_transition_matrix(P)
    return np.linalg.matrix_power(P, n)

if __name__ == '__main__':
    P = np.array([[0.7,0.2,0.1],[0.1,0.6,0.3],[0.4,0.2,0.4]])
    for n in [1,2,5,10,20]:
        print(f"{n}-step:\n", n_step_transition_probability(P,n))
    # plot evolution from state 0
    max_n = 20
    evo = np.array([n_step_transition_probability(P,n)[0] for n in range(1, max_n+1)])
    plt.figure(figsize=(8,5))
    for i in range(P.shape[0]):
        plt.plot(range(1,max_n+1), evo[:,i], marker='o', label=f'S0->S{i}')
    plt.legend(); plt.show()
