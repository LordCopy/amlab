import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, f

# Practical 7 — t and F distributions plotting

def plot_t_f():
    df_t = 10
    x_t = np.linspace(-5,5,500)
    pdf_t = t.pdf(x_t, df_t)

    dfn, dfd = 5, 20
    x_f = np.linspace(0,5,500)
    pdf_f = f.pdf(x_f, dfn, dfd)

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(x_t, pdf_t, 'g', label=f't (df={df_t})')
    plt.title('t Distribution')

    plt.subplot(1,2,2)
    plt.plot(x_f, pdf_f, 'c', label=f'F (dfn={dfn}, dfd={dfd})')
    plt.title('F Distribution')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_t_f()
