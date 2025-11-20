import numpy as np
import matplotlib.pyplot as plt

# Transition matrix
P = np.array([
    [0.7, 0.2, 0.1],
    [0.1, 0.6, 0.3],
    [0.4, 0.2, 0.4]
])

# Function to compute P^n
def n_step_transition(P, n):
    return np.linalg.matrix_power(P, n)

# Print matrices for some steps
steps = [1, 2, 5, 10, 20]
for n in steps:
    print( "Step : " , n , " transition probability matrix:")
    print(n_step_transition(P,n))

# ---- PLOT EVOLUTION FROM STATE S0 ----

max_n = 20  # total steps for plotting

# Compute probabilities of going from S0 to all states after n steps
evolution = np.array([n_step_transition(P, n)[0] for n in range(1, max_n + 1)])

# Plot
plt.figure(figsize=(8,5))

for i in range(P.shape[1]):  # for each state S0, S1, S2
    plt.plot(range(1, max_n + 1), evolution[:, i], marker='o', label=f"P(S0 -> S{i})")

plt.xlabel("Number of Steps (n)")
plt.ylabel("Probability")
plt.title("n-step Transition Probabilities from S0")
plt.xticks(range(1, max_n+1))
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
