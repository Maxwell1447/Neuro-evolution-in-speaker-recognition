import os

import torch
import matplotlib.pyplot as plt
from raw_audio_gender_classification.neat.rnn import load_data
from neat_local.nn import RecurrentNet
import neat
import numpy as np

batch_size = 50


def plot_gate_vs_signal(x, x_p, genome, conf):

    x_pt = x_p.transpose(0, 1)

    net = RecurrentNet.create(genome, conf, device="cpu", dtype=torch.float32)
    net.reset()
    gate = []
    # norm = torch.zeros(batch_size)
    for xi in x_pt:
        xo = net.activate(xi)  # batch_size x 2
        # score = xo[:, 1]
        confidence = xo[:, 0]
        gate.append(xo[:, 0].numpy())
        # contribution += score * confidence  # batch_size
        # norm += confidence

    # gate: t x batch_size(=1)
    gate = np.array(gate)

    assert gate.shape[1] == 1
    gate.flatten()
    gate = gate.repeat(512)


    x = x.numpy()
    norm = (x**2).sum() / gate.sum()
    
    plt.figure()
    plt.plot(x**2)
    plt.plot(gate/1.2)
    plt.show()


if __name__ == "__main__":

    winner = torch.load("best_genome_save")

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_preprocessed.cfg')
    config_ = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)

    _, test_loader_pp = load_data(batch_size=batch_size)
    _, test_loader = load_data(batch_size=batch_size, preprocessing=False)

    test_iter_pp = iter(test_loader_pp)
    test_iter = iter(test_loader)
    for _ in range(10):
        y, _ = next(test_iter)
        y_p, _ = next(test_iter_pp)

        plot_gate_vs_signal(y[0], y_p, winner, config_)




