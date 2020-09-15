import os
import neat

import backprop_neat
from anti_spoofing.eval_functions import evaluate_eer_acc
import torch
import numpy as np
import multiprocessing

import neat_local.visualization.visualize as visualize
from anti_spoofing.utils_ASV import make_visualize, show_stats
from anti_spoofing.data_loader import load_data
from anti_spoofing.eval_functions import eval_genome_eer, ProcessedASVEvaluator

"""
NEAT APPLIED TO ASVspoof 2019
"""


def run(config_file, n_gen):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: max number of generation
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
    stats_ = backprop_neat.StatisticsReporter()
    p.add_reporter(stats_)
    p.add_reporter(backprop_neat.Checkpointer(generation_interval=1000, time_interval_seconds=None))

    # Run for up to n_gen generations.
    if USE_DATASET:
        multi_evaluator = ProcessedASVEvaluator(multiprocessing.cpu_count(), eval_genome_eer,
                                                batch_increment=0, initial_batch_size=100, batch_generations=50)

    else:
        multi_evaluator = ProcessedASVEvaluator(multiprocessing.cpu_count(), eval_genome_eer, data=trainloader)

    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner_))

    return winner_, config_, stats_


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed.cfg')

    USE_DATASET = False

    trainloader, devloader, evalloader = load_data(batch_size=100, length=3 * 16000, num_train=40000,
                                                   num_test=40000, batch_size_test=5000, option="lfcc",
                                                   multi_proc=True, include_eval=True, return_dataset=USE_DATASET)

    eer_list = []
    accuracy_list = []

    eer_list_eval = []
    accuracy_list_eval = []

    winner_list = []

    for iterations in range(20):
        print("iterations number =", iterations)
        winner, config, stats = run(config_path, 500)
        visualize.plot_stats(stats, ylog=False, view=True)

        with torch.no_grad():
            eer, accuracy = evaluate_eer_acc(winner, config, devloader)
            eer_eval, accuracy_eval = evaluate_eer_acc(winner, config, evalloader)
        eer_list.append(eer)
        accuracy_list.append(accuracy)
        eer_list_eval.append(eer_eval)
        accuracy_list_eval.append(accuracy_eval)

    print("\n")
    print("equal error rate", eer_list)
    show_stats(np.array(eer_list))

    print("\n")
    print("equal error rate eval", eer_list_eval)
    show_stats(np.array(eer_list_eval))

    # make_visualize(winner, config, stats)
