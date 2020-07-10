import numpy as np
import tqdm

import neat_local.visualization.visualize as visualize
from anti_spoofing.metrics_utils import rocch2eer, rocch


SAMPLING_RATE = 16000


def sigmoid(x):
    """
    Implementation of the sigmoid function
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def leakyrelu(x, negative_slope: float = 0.01):
    """
    Implementation of the leakyrelu activation function
    :param x:
    :param negative_slope: slope for negative x
    :return:
    """
    return np.max(x, 0) + negative_slope * np.min(-x, 0)


def softmax(scores):
    """
    Implementation of softmax function for multi classification with cross entropy
    :param scores: scores related to the classes of the inputs
    :return: softmax of the scores
    """
    exp_score = np.exp(scores)
    return exp_score / np.sum(exp_score)


def whiten(sample_input):
    """
    normalize the input (expectancy is equal to 0, standard deviation to 1)
    the mask tells if the corresponding score should be taken into account
    :param sample_input: one audio file in a numpy format
    :return whitten_input: normalized input
    """
    whiten_input = sample_input - sample_input.mean()
    var = np.sqrt((whiten_input ** 2).mean())
    whiten_input *= 1 / var
    return whiten_input


def gate_average(recurrent_net, sample_input):
    """
    compute mask and the scores for a given input (audio file)
    the mask tells by how the corresponding score should be taken into account
    :param recurrent_net: network
    :param sample_input: one audio file in a numpy format
    :return mask, score
    """
    length = sample_input.size
    score, select = np.zeros(length), np.zeros(length)
    for i in range(length):
        select[i], score[i] = recurrent_net.activate([sample_input[i]])
    select = sigmoid(select)
    score = sigmoid(score)
    return select * score / np.sum(select)


def gate_activation(recurrent_net, sample_input):
    """
    compute mask and the scores for a given input (audio file)
    the mask tells if the corresponding score should be taken into account
    :param recurrent_net: network
    :param sample_input: one audio file in a numpy array format
    :return mask, score
    """
    length = sample_input.size
    score, select = np.zeros(length), np.zeros(length)
    for i in range(length):
        select[i], score[i] = recurrent_net.activate([sample_input[i]])
    mask = (select > 0.5)
    return mask, score


def gate_activation_ce(recurrent_net, sample_input):
    """
    compute mask and the scores for the several classes given input (audio file)
    the mask tells if the corresponding score should be taken into account
    :param recurrent_net: network
    :param sample_input: one audio file in a numpy array format
    :return mask, score
    """
    length = sample_input.size
    scores, select = np.zeros([length, 7]), np.zeros(length)
    for i in range(length):
        outputs = recurrent_net.activate([sample_input[i]])
        select[i], scores[i] = outputs[0], outputs[1:]
    mask = (select > 0.5)
    return mask, scores


def gate_activation_tensor(recurrent_net, sample_input):
    """
    compute mask and the scores for a given input (audio file)
    the mask tells if the corresponding score should be taken into account
    :param recurrent_net: network
    :param sample_input: one audio file in a tensor format
    :return mask, score
    """
    score, select = np.zeros(len(sample_input)), np.zeros(len(sample_input))
    for (i, xi) in enumerate(sample_input):
        select[i], score[i] = recurrent_net.activate([xi.item()])
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
        sample_input, output = data[0], data[1]
        sample_input = whiten(sample_input)
        mask, score = gate_activation(net, sample_input)
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


def evaluate_acc_eer(net, data_loader):
    """
    compute the eer equal error rate and the accuracy
    :param net: network
    :param data_loader: test dataset, contains audio files in a numpy array format
    :return eer and accuracy
    """
    correct = 0
    total = 0
    net.reset()
    target_scores = []
    non_target_scores = []
    for data in tqdm(data_loader):
        sample_input, output = data[0], data[1]
        sample_input = whiten(sample_input)
        mask, score = gate_activation(net, sample_input)
        selected_score = score[mask]
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        total += 1
        correct += ((xo > 0.5) == output)
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return float(correct) / total, eer


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

    node_names = {-1: "input", 0: "gate", 1: "bonafide speech", 2: "Wavenet vocoder", 3: "Conventional vocoder WORLD",
                  4: "Conventional vocoder MERLIN", 5: "Unit selection system MaryTTS",
                  6: "Voice conversion using neural networks", 7: "transform function-based voice conversion"}

    visualize.draw_net(config_, winner_, True, node_names=node_names)
    visualize.plot_stats(stats_, ylog=False, view=True)
    visualize.plot_species(stats_, view=True)


