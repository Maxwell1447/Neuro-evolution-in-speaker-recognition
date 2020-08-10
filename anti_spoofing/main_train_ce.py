import os
import neat
from neat_local.nn.recurrent_net import RecurrentNet
import torch
from torch import sigmoid
import numpy as np
import multiprocessing
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from anti_spoofing.utils_ASV import make_visualize
from anti_spoofing.data_loader import load_metadata
from anti_spoofing.eval_funtion_ce import eval_genome_ce, ProcessedASVEvaluator
from anti_spoofing.metrics_utils import rocch2eer, rocch

"""
NEAT APPLIED TO ASVspoof 2019
"""


def run(config_file, n_gen):
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

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config_)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats_ = neat.StatisticsReporter()
    p.add_reporter(stats_)
    p.add_reporter(neat.Checkpointer(generation_interval=1000, time_interval_seconds=None))

    # Run for up to n_gen generations.
    multi_evaluator = ProcessedASVEvaluator(multiprocessing.cpu_count(), eval_genome_ce, trainloader)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner_))

    return winner_, config_, stats_


def evaluate(g, conf, data):
    """
    returns the equal error rate and the accuracy
    """
    data_iter = iter(data)

    target_scores = []
    non_target_scores = []

    jitter = 1e-8
    correct = 0
    correct_anti_spoofing = 0
    total = len(data)

    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)

    for i in range(total):
        net.reset()
        batch = next(data_iter)
        input, _, output = batch
        input = input[0]
        xo = net.activate(input)  # batch_size x 8
        score = xo[:, 1:].reshape(torch.Size((7, -1)))
        confidence = sigmoid(xo[:, 0])
        contribution = (score * confidence).sum(axis=1) / (jitter + confidence).sum()
        if output == 0:
            target_scores.append(contribution[0].item())
            correct_anti_spoofing += (contribution.argmax() == 0).item()
        else:
            non_target_scores.append(contribution[0].item())
            correct_anti_spoofing += (contribution.argmax() > 0).item()

        correct += (contribution.argmax() == output).item()

    accuracy = correct / total
    anti_spoofing_accuracy = correct_anti_spoofing / total

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer, accuracy, anti_spoofing_accuracy


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed_ce.cfg')

    trainloader, testloader = load_metadata(batch_size=100, length=3 * 16000, num_train=10000)

    eer_list = []
    accuracy_list = []
    anti_spoofing_accuracy_list = []
    for iterations in range(1):
        print(iterations)
        print(eer_list)
        winner, config, stats = run(config_path, 1000)

        eer, accuracy, anti_spoofing_accuracy = evaluate(winner, config, testloader)
        eer_list.append(eer)
        accuracy_list.append(accuracy)
        anti_spoofing_accuracy_list.append(anti_spoofing_accuracy)

    print("\n")
    print("equal error rate", eer_list)
    print("accuracy", accuracy_list)
    print("accuracy", anti_spoofing_accuracy_list)

    print("\n")

    array_eer = np.array(eer_list)

    print("min =", array_eer.min())
    print("max =", array_eer.max())
    print("median =", np.median(array_eer))
    print("average =", array_eer.mean())
    print("std =", array_eer.std())
    make_visualize(winner, config, stats)
