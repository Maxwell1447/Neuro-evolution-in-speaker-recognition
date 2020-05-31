
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
