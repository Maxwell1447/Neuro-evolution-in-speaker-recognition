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
from anti_spoofing.data_loader import load_data
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
    # p.add_reporter(neat.Checkpointer(generation_interval=100, time_interval_seconds=None))

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
    total = 0

    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset()
    for i in range(len(data)):
        batch = next(data_iter)
        input, _, output = batch
        input = input[0]
        xo = net.activate(input)  # batch_size x 8
        score = xo[:, 1:].reshape(torch.Size((7, -1)))
        confidence = sigmoid(xo[:, 0])
        contribution = (score * confidence).sum(axis=1) / (jitter + confidence).sum()
        if output == 0:
            target_scores.append(contribution[0].item())
        else:
            non_target_scores.append(contribution[0].item())

        correct += (contribution.argmax() == output).item()
        total += 1
    accuracy = correct / total

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer, accuracy


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed_ce.cfg')

    trainloader, testloader = load_data(batch_size=100, length=3 * 16000, num_train=10000, num_test=10000)

    eer_list = []
    accuracy_list = []
    for iterations in range(20):
        winner, config, stats = run(config_path, 500)

        eer, accuracy = evaluate(winner, config, testloader)
        eer_list.append(eer)
        accuracy_list.append(accuracy)

    print("\n")
    print("equal error rate", eer_list)
    print("accuracy", accuracy_list)

    # make_visualize(winner, config, stats)
