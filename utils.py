import numpy as np


def smooth(values, momentum=0.99):

    mu = momentum
    w = 1
    s = 0

    smoothed = []
    for v in values:
        s = s * mu + v
        smoothed.append(s/w)
        w = 1 + mu * w

    return smoothed


def smooth2(values, window=10):

    vals = np.full(values.size + window - 1, values[0])
    vals[window-1:] = values
    
    w = np.ones(window) / window

    return np.convolve(vals, w, mode='valid')