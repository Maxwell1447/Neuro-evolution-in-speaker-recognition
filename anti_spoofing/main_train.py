import os
import neat
import numpy as np
import multiprocessing
from tqdm import tqdm
from anti_spoofing.utils_ASV import make_visualize
from anti_spoofing.data_loader import load_data
from anti_spoofing.eval_functions import eval_genome_bce, ProcessedASVEvaluator

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
    p.add_reporter(neat.Checkpointer(generation_interval=100, time_interval_seconds=None))

    # Run for up to n_gen generations.
    multi_evaluator = ProcessedASVEvaluator(multiprocessing.cpu_count(), eval_genome_bce, trainloader)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    return winner_, config_, stats_


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed.cfg')

    trainloader, testloader = load_data(batch_size=50, length=3*16000, num_train=1000)

    winner, config, stats = run(config_path, 1000)
    make_visualize(winner, config, stats)
