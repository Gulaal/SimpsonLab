import matplotlib.pyplot as plt
import numpy as np

def draw(c, d, t):
    x = np.linspace(c, d, 100)
    y = np.exp(-((t/(1+x**2)) +0.001*x))

    plt.plot(x, y)
    plt.show()