import numpy as np
import matplotlib.pyplot as plt

def signal(n=100, f=0.01, phi=0., magnitude=1.):

    idx = np.arange(n)
    return magnitude * np.sin(2 * np.pi * f * idx + phi)


plt.plot(signal(n=1000))
plt.plot(signal(n=1000, f=0.002))