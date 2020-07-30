"""
Visualize the behavior of the gate w.r.t. the original signal.
"""

import os

import torch
import matplotlib.pyplot as plt
from raw_audio_gender_classification.neat.rnn import load_data
from neat_local.nn import RecurrentNet
import neat
import numpy as np
from preprocessing_tools.preprocessing import preprocess
import librosa
from scipy.fftpack import fft, rfft

batch_size = 50


def plot_gate_vs_signal(x, x_p, genome, conf):
    """
    x: numpy array corresponding to the raw audio signal
    x_p: preprocessed signal
    genome: genome used for the analysis (best genome)
    conf: config file

    plot the squared signal w.r.t. the windowed-gate signal
    """

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
    # norm = (x**2).sum() / gate.sum()

    plt.figure()
    plt.plot(x**2)
    plt.plot(gate/2.)
    plt.show()


def plot_gate_vs_signal_linear(x, x_p, model):
    """
        x: numpy array corresponding to the raw audio signal
        x_p: preprocessed signal
        model: linear model

        plot the squared signal w.r.t. the windowed-gate signal
        """

    x_pt = x_p.transpose(0, 1)

    # norm = torch.zeros(batch_size)

    with torch.no_grad():
        print(model.fc1.weight)
        score, gate = model.score_gate(x_pt)  # t x batch_size

        # gate: t x batch_size(=1)
        gate = gate.numpy()

    assert gate.shape[1] == 1
    gate.flatten()
    gate = gate.repeat(512)

    x = x.numpy()[0]

    filtered_signal = gate[:len(x)] * x
    # norm = (x**2).sum() / gate.sum()
    plt.figure()
    plt.plot(x)
    plt.plot(filtered_signal)
    plt.plot(gate)
    plt.show()


def plot_gated_fft(x, x_p, genome, conf, t):
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

    filtered_signal = gate[:len(x)] * x.numpy()

    # plt.figure()
    # plt.plot(x)
    # plt.plot(filtered_signal)
    # plt.plot(gate)
    # plt.show()

    print(t)

    # x_fft = rfft(x.numpy())
    #
    # plt.figure()
    # plt.title("signal female" if t else "signal male")
    # plt.plot(x_fft)
    # plt.show()

    x_fft = rfft(filtered_signal)

    plt.figure()
    plt.title("filtered signal female" if t else "filtered signal male")
    plt.plot(x_fft[:len(x_fft)//94])
    plt.show()


def test_correspondence(s, s_p):
    print(s.shape, s_p.shape)
    s_p2 = preprocess(s, option="mfcc")
    print(np.all(np.abs(s_p - s_p2) < 1e-3))
    print()


if __name__ == "__main__":

    winner = torch.load("best_genome_save")

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat_preprocessed.cfg')
    config_ = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_path)

    # Used on the test set
    _, test_loader_pp = load_data(batch_size=batch_size)
    _, test_loader = load_data(batch_size=batch_size, preprocessing=False)

    test_iter_pp = iter(test_loader_pp)
    test_iter = iter(test_loader)
    for i in range(150):  # number of audio files taken

        y, t = next(test_iter)
        y_p, _ = next(test_iter_pp)

        if i % 17 == 0:
            # plot_gate_vs_signal(y[0], y_p, winner, config_)
            # plot_gate_vs_signal_linear(y, y_p, torch.load("linear_SGD.pkl"))
            plot_gated_fft(y[0], y_p, winner, config_, t[0])
