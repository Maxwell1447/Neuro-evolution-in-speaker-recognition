import numpy as np
import tqdm

import neat_local.visualization.visualize as visualize
from anti_spoofing.metrics_utils import rocch2eer, rocch


def whiten(input):
    """
    normalize the input (expectancy is equal to 0, standard deviation to 1)
    the mask tells if the corresponding score should be taken into account
    :param input: one audio file in a numpy format
    :return whitten_input: normalized input
    """
    whiten_input = input - input.mean()
    var = np.sqrt((whiten_input ** 2).mean())
    whiten_input *= 1 / var
    return whiten_input


def gate_average(recurrent_net, input):
    """
    compute mask and the scores for a given input (audio file)
    the mask tells by how the corresponding score should be taken into account
    :param recurrent_net: network
    :param input: one audio file in a numpy format
    :return mask, score
    """
    length = input.size
    score, select = np.zeros(length), np.zeros(length)
    for i in range(length):
        select[i], score[i] = recurrent_net.activate([input[i]])
    return select, score


def gate_activation(recurrent_net, input):
    """
    compute mask and the scores for a given input (audio file)
    the mask tells if the corresponding score should be taken into account
    :param recurrent_net: network
    :param input: one audio file in a numpy array format
    :return mask, score
    """
    length = input.size
    score, select = np.zeros(length), np.zeros(length)
    for i in range(length):
        select[i], score[i] = recurrent_net.activate([input[i]])
    mask = (select > 0.5)
    return mask, score


def evaluate(net, data_loader):
    """
    compute the eer equal error rate
    :param net: network
    :param data_loader: test dataset, contains audio files in a numpy array format
    :return eer
    """
    net.reset()
    target_scores = []
    non_target_scores = []
    for data in tqdm(data_loader):
        inputs, output = data[0], data[1]
        inputs = whiten(inputs)
        mask, score = gate_activation(net, inputs)
        selected_score = score[mask]
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer


def make_visualize(winner_, config_, stats_):
    """
    Plot and draw:
        - the graph of the topology
        - the fitness evolution over generations
        - the speciation evolution over generations
    :param winner_: genome
    :param config_: configuration file
    :param stats_: statistics from reporter
    :return None
    """

    node_names = {-1: "input", 1: "score", 0: "gate"}

    visualize.draw_net(config_, winner_, True, node_names=node_names)
    visualize.plot_stats(stats_, ylog=False, view=True)
    visualize.plot_species(stats_, view=True)
