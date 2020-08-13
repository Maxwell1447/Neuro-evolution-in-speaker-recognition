import os
import neat

from anti_spoofing.eval_functions import feed_and_predict, evaluate_eer_acc
from neat_local.nn.recurrent_net import RecurrentNet
import torch
import numpy as np
import multiprocessing
from tqdm import tqdm

from anti_spoofing.utils_ASV import make_visualize
from anti_spoofing.data_loader import load_data
from anti_spoofing.eval_function_eoc import eval_genome_eoc, eval_eer_gc, ProcessedASVEvaluatorEocGc


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
    multi_evaluator = ProcessedASVEvaluatorEocGc(multiprocessing.cpu_count(), eval_genome_eoc,
                                               data=trainloader, pop=config_.pop_size, config=config_,
                                               gc_eval=eval_eer_gc, validation_data=testloader)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    gc = multi_evaluator.gc

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner_))

    return gc, winner_, config_, stats_


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed.cfg')

    trainloader, testloader = load_data(batch_size=100, length=3*16000, num_train=10000,
                                        batch_size_test=1000, option="lfcc", multi_proc=True,)

    eer_list_gc = []
    accuracy_list_gc = []
    eer_list = []
    accuracy_list = []
    for iterations in range(10):
        print(iterations)
        print(eer_list)
        gc, winner, config, stats = run(config_path, 200)

        with torch.no_grad():
            eer, accuracy = evaluate_eer_acc(winner, config, testloader)
            eer_gc, accuracy_gc = evaluate_eer_acc(gc, config, testloader)
        eer_list.append(eer)
        accuracy_list.append(accuracy)
        eer_list_gc.append(eer_gc)
        accuracy_list_gc.append(accuracy_gc)

    print("\n")
    print("equal error rate", eer_list)
    print("accuracy", accuracy_list)

    print("\n")

    array_eer = np.array(eer_list)

    print("min =", array_eer.min())
    print("max =", array_eer.max())
    print("median =", np.median(array_eer))
    print("average =", array_eer.mean())
    print("std =", array_eer.std())
    # make_visualize(winner, config, stats)

    array_eer_gc = np.array(eer_list_gc)

    print("min =", array_eer_gc.min())
    print("max =", array_eer_gc.max())
    print("median =", np.median(array_eer_gc))
    print("average =", array_eer_gc.mean())
    print("std =", array_eer_gc.std())
