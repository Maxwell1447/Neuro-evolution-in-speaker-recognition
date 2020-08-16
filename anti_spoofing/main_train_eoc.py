import os
import neat

from anti_spoofing.eval_functions import feed_and_predict, evaluate_eer_acc
from neat_local.nn.recurrent_net import RecurrentNet
import torch
import numpy as np
import multiprocessing
from tqdm import tqdm

from anti_spoofing.utils_ASV import make_visualize, show_stats
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
                                                 gc_eval=eval_eer_gc, validation_data=devloader)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    gc = multi_evaluator.gc
    generations_gc = multi_evaluator.app_gc
    print("Grand champion appeared at generation ", generations_gc)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner_))

    return gc, winner_, config_, stats_, generations_gc


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed.cfg')

    trainloader, devloader, evalloader = load_data(batch_size=100, length=3 * 16000, num_train=30000,
                                                   num_test=40000, batch_size_test=24844, option="lfcc",
                                                   multi_proc=True, include_eval=True)

    eer_list_gc = []
    accuracy_list_gc = []
    eer_list = []
    accuracy_list = []

    eer_list_gc_eval = []
    accuracy_list_gc_eval = []
    eer_list_eval = []
    accuracy_list_eval = []

    winner_list = []
    gc_list = []
    generation_gc_list = []

    for iterations in range(20):
        print("iterations number =", iterations)
        gc, winner, config, stats, gen_gc = run(config_path, 100)

        with torch.no_grad():
            eer, accuracy = evaluate_eer_acc(winner, config, devloader)
            eer_eval, accuracy_eval = evaluate_eer_acc(winner, config, evalloader)
            eer_gc, accuracy_gc = evaluate_eer_acc(gc, config, devloader)
            eer_gc_eval, accuracy_gc_eval = evaluate_eer_acc(gc, config, evalloader)
        eer_list.append(eer)
        accuracy_list.append(accuracy)
        eer_list_gc.append(eer_gc)
        accuracy_list_gc.append(accuracy_gc)
        eer_list_eval.append(eer_eval)
        accuracy_list_eval.append(accuracy_eval)
        eer_list_gc_eval.append(eer_gc_eval)
        accuracy_list_gc_eval.append(accuracy_gc_eval)
        generation_gc_list.append(gen_gc)

    print("\n")
    print("equal error rate", eer_list)
    show_stats(np.array(eer_list))

    print("\n")
    print("equal error rate gc", eer_list_gc)
    show_stats(np.array(eer_list_gc))

    print("\n")
    print("equal error rate eval", eer_list_eval)
    show_stats(np.array(eer_list_eval))

    print("\n")
    print("equal error rate gc eval", eer_list_gc_eval)
    show_stats(np.array(eer_list_gc_eval))

    print("\n")
    print("Grand champions appeared at generations", generation_gc_list)
    show_stats(np.array(generation_gc_list))

    make_visualize(winner, config, stats)
    make_visualize(gc, config, stats)
