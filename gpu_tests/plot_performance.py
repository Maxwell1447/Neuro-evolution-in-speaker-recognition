import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# PARAMETERS MANUALLY DEDUCED
a_v = 1.3 * 10**-7
b_v = 1.0 * 10**-6

a_c = 3.0 * 10**-10
b_c1 = 1.5 * 10**-5
b_c2 = 5.6 * 10**-12
c_c = 1.0 * 10**-4

b_g1 = b_c1
b_g2 = b_c2
c_g = 7.5 * 10**-4
d_g = 2.0

cmap_light = ListedColormap(['#FF7777', '#77FF77', '#7777FF'])


def time_v(C, I):
    """
    Feed time of "vanilla"
    """
    return a_v * C * I + b_v * C


def time_c(C, I):
    """
    Feed time of "cpu"
    """
    return (a_c * C + c_c) * I + (b_c1 + b_c2 * C) * C


def time_g(C, I):
    """
    Feed time of "cuda"
    """
    return c_g * I + (b_g1 + b_g2 * C) * C


def evaluate(ravel):
    """
    Meshgrid evaluation with:
    * 0 --> "vanilla" is best
    * 1 --> "cpu" is best
    * 2 --> "cuda" is best
    """
    Z = np.zeros(ravel.shape[0])
    for i, (C, I) in enumerate(ravel):
        TV, TC, TG = time_v(C, I), time_c(C, I), time_g(C, I)
        if TV < TC and TV < TG:
            Z[i] = 0
        elif TC < TV and TC < TG:
            Z[i] = 1
        else:
            Z[i] = 2
    return Z


if __name__ == "__main__":

    C_min, C_max = 10**2, 10**8
    I_min, I_max = 1., 10**4
    r = (C_max / C_min) ** (1. / 500)
    xx = np.arange(500)
    xx = C_min * r**xx
    xx, yy = np.meshgrid(xx,
                         np.linspace(I_min, I_max, 500))

    Z = evaluate(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.xscale("log")
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    proxy = [plt.Rectangle((0, 0), 1, 1, fc=color)
             for color in ['#FF7777', '#77FF77', '#7777FF']]

    plt.legend(proxy, ["Vanilla", "CPU", "GPU"])

    plt.xlabel("C")
    plt.ylabel("I")
    plt.title("best timing for a couple (I, C)")
    plt.show()


