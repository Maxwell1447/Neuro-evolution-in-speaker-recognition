import os
import neat
from neat_local.nn.recurrent_net import RecurrentNet
import torch
from torch import sigmoid
import numpy as np
import multiprocessing
from tqdm import tqdm

from anti_spoofing.utils_ASV import make_visualize
from anti_spoofing.data_loader import load_data, load_data_cqcc
from anti_spoofing.eval_functions import eval_genome_bce, eval_genome_eer, ProcessedASVEvaluator
from anti_spoofing.eval_function_eoc import eval_genome_eoc, ProcessedASVEvaluatorEoc, ProcessedASVEvaluatorEocGc
from anti_spoofing.metrics_utils import rocch2eer, rocch
from anti_spoofing.constants import *
from neat_local.scheduler import ExponentialScheduler, OnPlateauScheduler, \
    ImpulseScheduler, SineScheduler, MutateScheduler
from neat_local.reporters import ComplexityReporter
import os
import sys

"""
NEAT APPLIED TO ASVspoof 2019
"""


if sys.platform.find("win") >= 0:
    DATA_ROOT = './data'
else:
    DATA_ROOT = os.path.join("..", "..", "..", "speechmaterials", "databases", "ASVspoof")


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
    # sine_scheduler = SineScheduler(config_, period=500, final_values={
    #     "node_add_prob": 0.0,
    #     "conn_add_prob": 0.0,
    #     "node_delete_prob": 0.0,
    #     "conn_delete_prob": 0.0
    # })
    # mutate_scheduler = MutateScheduler(parameters=["node_add_prob", "conn_add_prob",
    #                                                "node_delete_prob", "conn_delete_prob"],
    #                                    patience=2, momentum=0.99)
    # scheduler = ExponentialScheduler(semi_gen=2000, final_values={
    #     "node_add_prob": 0.,
    #     "conn_add_prob": 0.
    # })
    # scheduler2 = ExponentialScheduler(semi_gen=3000, final_values={
    #     "node_delete_prob": 0.,
    #     "conn_delete_prob": 0.
    # })
    # impulse_scheduler = ImpulseScheduler(parameters=["node_add_prob", "conn_add_prob",
    #                                                  "node_delete_prob", "conn_delete_prob"],
    #                                      verbose=1, patience=10, impulse_factor=2., momentum=0.99, monitor=True)
    # scheduler = OnPlateauScheduler(parameters=["node_add_prob", "conn_add_prob",
    #                                            "node_delete_prob", "conn_delete_prob"],
    #                                verbose=1, patience=10, factor=0.995, momentum=0.99)
    # p.add_reporter(mutate_scheduler)
    # p.add_reporter(sine_scheduler)
    # p.add_reporter(scheduler2)
    # p.add_reporter(impulse_scheduler)
    complexity_reporter = ComplexityReporter()
    p.add_reporter(complexity_reporter)

    # Run for up to n_gen generations.
    multi_evaluator = ProcessedASVEvaluator(multiprocessing.cpu_count(), eval_genome_bce, trainloader)
    # multi_evaluator = ProcessedASVEvaluatorEoc(multiprocessing.cpu_count(), eval_genome_eoc, trainloader,
    #                                            getattr(config_, "pop_size"))

    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    complexity_reporter.display()
    # impulse_scheduler.display()
    # sine_scheduler.display()
    # mutate_scheduler.display()

    print("run finished")

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
    for _ in tqdm(range(len(data))):
        net.reset()
        batch = next(data_iter)
        input, output = batch
        input = input[0]
        xo = net.activate(input)
        score = xo[:, 1]
        confidence = xo[:, 0]
        contribution = (score * confidence).sum() / (jitter + confidence.sum())

        if output == 1:
            target_scores.append(contribution)
        else:
            non_target_scores.append(contribution)

        correct += ((contribution > 0.5) == output).item()
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
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed.cfg')

    if OPTION == "cqcc":
        trainloader, testloader = load_data_cqcc(batch_size=100, num_train=10000, num_test=10000, balanced=True)
    else:
        trainloader, testloader = load_data(batch_size=100, length=3 * 16000, num_train=10000, custom_path=DATA_ROOT)

    winner, config, stats = run(config_path, 100)

    eer, accuracy = evaluate(winner, config, testloader)

    print("\n")
    print("equal error rate", eer)
    print("accuracy", accuracy)

    make_visualize(winner, config, stats)
