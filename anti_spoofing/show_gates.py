from __future__ import print_function
import os
import neat
import neat_local.visualization.visualize as visualize
import numpy as np
import random as rd

from tqdm import tqdm

from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.metrics_utils import rocch2eer, rocch

path = "neat-checkpoint-39"


n_processes = 6  # multiprocessing.cpu_count()

dev_border = [0, 2548, 6264, 9980, 13696, 17412, 21128, 22296]
index_test = []
for i in range(len(dev_border) - 1):
    index_test += rd.sample([k for k in range(dev_border[i], dev_border[i + 1])], 1)

test_loader = ASVDataset(None, is_train=False, is_eval=False, index_list=index_test)


def whiten(single_input):
    whiten_input = single_input - single_input.mean()
    var = np.sqrt((whiten_input ** 2).mean())
    whiten_input *= 1 / var
    return whiten_input


def gate_activation(recurrent_net, inputs):
    length = inputs.size
    score, select = np.zeros(length), np.zeros(length)
    for i in range(length):
        select[i], score[i] = recurrent_net.activate([inputs[i]])
    mask = (select > 0.5)
    return mask, score


def evaluate(net, data_loader):
    net.reset()
    target_scores = []
    non_target_scores = []
    gates = []
    for data in data_loader:
        inputs, output = data[0], data[1]
        inputs = whiten(inputs)
        mask, score = gate_activation(net, inputs)
        selected_score = score[mask]
        gates.append(mask)
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

    return np.array(gates), eer


def run(config_file, path):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genontype (winner), the configs, the stats of the run and the accuracy on the testing set
    """
    # Load configuration.
    config_ = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # load saved population
    p = neat.Checkpointer.restore_checkpoint(path)

    genomes = p.population

    gates = []

    for genome_id in tqdm(genomes):
        net = neat.nn.RecurrentNetwork.create(genomes[genome_id], config_)
        gate, eer = evaluate(net, test_loader)
        gates.append(gate)

    return np.array(gates)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')
    gates = run(config_path, path)
