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
from anti_spoofing.eval_functions import eval_genome_bce, eval_genome_eer, ProcessedASVEvaluator, feed_and_predict, \
    evaluate_eer_acc
from anti_spoofing.eval_function_eoc import eval_genome_eoc, ProcessedASVEvaluatorEoc, ProcessedASVEvaluatorEocGc
from anti_spoofing.metrics_utils import rocch2eer, rocch
from anti_spoofing.constants import *
from neat_local.scheduler import ExponentialScheduler, OnPlateauScheduler, \
    ImpulseScheduler, SineScheduler, MutateScheduler
from neat_local.reporters import ComplexityReporter
from anti_spoofing.select_best import get_true_winner
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
    :return: the best genotype (winner), the configs, the stats of the run and the accuracy on the testing set
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
    # p.add_reporter(sine_scheduler)
    # mutate_scheduler = MutateScheduler(parameters=["node_add_prob", "conn_add_prob",
    #                                                "node_delete_prob", "conn_delete_prob"],
    #                                    patience=2, momentum=0.99)
    # p.add_reporter(mutate_scheduler)

    scheduler = ExponentialScheduler(semi_gen=2000, final_values={
        "node_add_prob": 0.,
        "conn_add_prob": 0.
    })
    p.add_reporter(scheduler)
    scheduler2 = ExponentialScheduler(semi_gen=3000, final_values={
        "node_delete_prob": 0.,
        "conn_delete_prob": 0.
    })
    p.add_reporter(scheduler2)
    # impulse_scheduler = ImpulseScheduler(parameters=["node_add_prob", "conn_add_prob",
    #                                                  "node_delete_prob", "conn_delete_prob"],
    #                                      verbose=1, patience=10, impulse_factor=2., momentum=0.99, monitor=True)
    # scheduler = OnPlateauScheduler(parameters=["node_add_prob", "conn_add_prob",
    #                                            "node_delete_prob", "conn_delete_prob"],
    #                                verbose=1, patience=10, factor=0.995, momentum=0.99)

    # p.add_reporter(impulse_scheduler)
    complexity_reporter = ComplexityReporter()
    p.add_reporter(complexity_reporter)

    # Run for up to n_gen generations.
    # multi_evaluator = ProcessedASVEvaluator(multiprocessing.cpu_count(), eval_genome_bce, trainloader)
    multi_evaluator = ProcessedASVEvaluatorEoc(multiprocessing.cpu_count(), eval_genome_eoc, trainloader,
                                               getattr(config_, "pop_size"))

    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # winner_ = get_true_winner(config_, p.population, trainloader, max_batch=10)

    # complexity_reporter.display()

    # impulse_scheduler.display()
    # sine_scheduler.display()
    # mutate_scheduler.display()

    print("run finished")

    return winner_, config_, stats_


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed.cfg')

    if OPTION == "cqcc":
        trainloader, devloader = load_data_cqcc(batch_size=100, num_train=10000, num_test=10000, balanced=True)
        evalloader = None
    else:
        trainloader, devloader, evalloader = load_data(batch_size=100, length=3 * 16000, num_train=10000,
                                                       custom_path=DATA_ROOT, multi_proc=True, balanced=True,
                                                       batch_size_test=100, include_eval=True)

    dev_eer_list = []
    dev_accuracy_list = []
    eval_eer_list = []
    eval_accuracy_list = []
    for i in range(20):
        print(i)
        print(dev_eer_list)
        winner, config, stats = run(config_path, 200)

        eer, accuracy = evaluate_eer_acc(winner, config, devloader)
        dev_eer_list.append(eer)
        dev_accuracy_list.append(accuracy)

        eer, accuracy = evaluate_eer_acc(winner, config, evalloader)
        eval_eer_list.append(eer)
        eval_accuracy_list.append(accuracy)

        if i == 0:
            make_visualize(winner, config, stats, topology=False)

    print("\n")
    print("DEV equal error rate", dev_eer_list)
    print("accuracy", dev_accuracy_list)

    print("\n")

    array_eer = np.array(dev_eer_list)

    print("min =", array_eer.min())
    print("max =", array_eer.max())
    print("median =", np.median(array_eer))
    print("average =", array_eer.mean())
    print("std =", array_eer.std())

    print("\n")
    print("EVAL equal error rate", eval_eer_list)
    print("accuracy", eval_accuracy_list)

    print("\n")

    array_eer = np.array(eval_eer_list)

    print("min =", array_eer.min())
    print("max =", array_eer.max())
    print("median =", np.median(array_eer))
    print("average =", array_eer.mean())
    print("std =", array_eer.std())
