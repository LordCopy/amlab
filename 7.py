import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, f

# Practical 7: Generate t & F Distribution

def plot_t_f():
    # t-distribution
    df_t = 10
    x_t = np.linspace(-5, 5, 500)
    pdf_t = t.pdf(x_t, df_t)

    # F-distribution
    dfn, dfd = 5, 20
    x_f = np.linspace(0, 5, 500)
    pdf_f = f.pdf(x_f, dfn, dfd)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_t, pdf_t, 'g', label=f't-distribution (df={df_t})')
    plt.title('t Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x_f, pdf_f, 'c', label=f'F-distribution (dfn={dfn}, dfd={dfd})')
    plt.title('F Distribution')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_t_f()
