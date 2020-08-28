import os
import shutil
from torch.utils.tensorboard import SummaryWriter

from neat_local.nn.recurrent_net import RecurrentNet
import torch
from torch import sigmoid
import numpy as np
import multiprocessing
from tqdm import tqdm
import random

from anti_spoofing.utils_ASV import make_visualize
from anti_spoofing.data_loader import load_data, load_data_cqcc
from anti_spoofing.eval_functions import eval_genome_bce, eval_genome_eer, ProcessedASVEvaluator, feed_and_predict, \
    evaluate_eer_acc
from anti_spoofing.eval_function_eoc import eval_genome_eoc, ProcessedASVEvaluatorEoc, ProcessedASVEvaluatorEocGc, \
    quantified_eval_genome_eoc, double_quantified_eval_genome_eoc
from anti_spoofing.metrics_utils import rocch2eer, rocch
from anti_spoofing.constants import *
from neat_local.scheduler import *
from neat_local.reporters import ComplexityReporter, EERReporter, StdOutReporter, WriterReporter
from anti_spoofing.select_best import get_true_winner
import os
import sys
import backprop_neat

"""
NEAT APPLIED TO ASVspoof 2019
"""

if sys.platform.find("win") >= 0:
    DATA_ROOT = './data'
else:
    DATA_ROOT = os.path.join("..", "..", "..", "speechmaterials", "databases", "ASVspoof")

backprop = False
USE_DATASET = False
USE_GATE = True
KEEP_FROM = 0

if backprop:
    import backprop_neat as neat
else:
    import neat


def reporter_addition(p, config_):
    displayable = []
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(StdOutReporter(True))
    stats_ = backprop_neat.StatisticsReporter()
    p.add_reporter(stats_)
    p.add_reporter(backprop_neat.Checkpointer(generation_interval=1000, time_interval_seconds=None))

    written_params = ("conn_add_prob", "backprop") if backprop else ("conn_add_prob",)
    p.add_reporter(WriterReporter(writer, written_params))

    # eer_acc_reporter = EERReporter(devloader, period=2)
    # p.add_reporter(eer_acc_reporter)

    start = 200

    # adaptive_backprop_scheduler = AdaptiveBackpropScheduler(config_, patience=40, semi_gen=20,
    #                                                         monitor=True, start=start, patience_before_backprop=50,
    #                                                         privilege=100,
    #                                                         values=[
    #                                                             "node_add_prob",
    #                                                             "conn_add_prob",
    #                                                             "node_delete_prob",
    #                                                             "conn_delete_prob",
    #                                                             "bias_mutate_rate",
    #                                                             "weight_mutate_rate",
    #                                                             "bias_replace_rate",
    #                                                             "weight_replace_rate"
    #                                                         ])
    # p.add_reporter(adaptive_backprop_scheduler)

    for param in ["node_add_prob", "conn_add_prob", "node_delete_prob", "conn_delete_prob",
                  "bias_mutate_rate", "weight_mutate_rate", "bias_replace_rate", "weight_replace_rate"]:
        print(getattr(config_.genome_config, param))
        exit(8)

    early_exploration_scheduler = EarlyExplorationScheduler(config_, duration=start,
                                                            values={
                                                                "node_add_prob": 0.5,
                                                                "conn_add_prob": 0.8,
                                                                "node_delete_prob": 0.1,
                                                                "conn_delete_prob": 0.1,
                                                                "bias_mutate_rate": 0.8,
                                                                "weight_mutate_rate": 0.8,
                                                                "bias_replace_rate": 0.1,
                                                                "weight_replace_rate": 0.1,
                                                            },
                                                            reset=True,
                                                            monitor=True,
                                                            verbose=1)
    p.add_reporter(early_exploration_scheduler)

    squashed_sine_scheduler = SquashedSineScheduler(config_, offset=start,
                                                    period=150,
                                                    final_values={
                                                        "node_add_prob": 0.,
                                                        "conn_add_prob": 0.,
                                                        "node_delete_prob": 0.,
                                                        "conn_delete_prob": 0.,
                                                        "bias_mutate_rate": 0.,
                                                        "weight_mutate_rate": 0.,
                                                        "bias_replace_rate": 0.,
                                                        "weight_replace_rate": 0.,
                                                    },
                                                    monitor=True,
                                                    verbose=0,
                                                    alpha=3)
    p.add_reporter(squashed_sine_scheduler)

    p.add_reporter(DisableBackpropScheduler())

    complexity_reporter = ComplexityReporter()
    p.add_reporter(complexity_reporter)

    displayable.append(complexity_reporter)

    if backprop:
        displayable.append(early_exploration_scheduler)
        # displayable.append(adaptive_backprop_scheduler)

    return displayable, stats_


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
    if KEEP_FROM > 0:
        p, w = neat.Checkpointer.restore_checkpoint("neat-checkpoint_-{}".format(KEEP_FROM))
        p.add_reporter(WriterReporter(writer, params=w))
        stats_ = None
        displayable = []
        for reporter in p.reporters.reporters:
            if isinstance(reporter, neat.StatisticsReporter):
                stats_ = reporter
            elif isinstance(reporter, (EarlyExplorationScheduler, AdaptiveBackpropScheduler,
                                       ComplexityReporter)):
                displayable.append(reporter)
    else:
        p = neat.Population(config_)
        displayable, stats_ = reporter_addition(p, config_)

    # Run for up to n_gen generations.

    multi = backprop + (not backprop) * multiprocessing.cpu_count()

    if USE_DATASET:
        multi_evaluator = ProcessedASVEvaluator(multi, eval_genome_bce, train_data,
                                                batch_increment=0, initial_batch_size=100,
                                                backprop=backprop, use_gate=USE_GATE)
        # multi_evaluator = ProcessedASVEvaluatorEoc(multi, quantified_eval_genome_eoc,
        #                                            train_data,
        #                                            getattr(config_, "pop_size"),
        #                                            batch_increment=50, initial_batch_size=100, batch_generations=50,
        #                                            backprop=backprop, use_gate=USE_GATE)
    else:
        multi_evaluator = ProcessedASVEvaluator(multi, eval_genome_bce, train_data,
                                                use_gate=USE_GATE)
        # multi_evaluator = ProcessedASVEvaluatorEoc(multi, double_quantified_eval_genome_eoc,
        #                                            train_data,
        #                                            getattr(config_, "pop_size"), use_gate=USE_GATE)

    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # winner_ = get_true_winner(config_, p.population, trainloader, max_batch=10)

    for reporter in displayable:
        reporter.display()

    print("run finished")

    return winner_, config_, stats_


if __name__ == '__main__':

    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed{}.cfg'.format('_backprop' if backprop else '_long'))

    print("dir:   ", config_path)

    if OPTION == "cqcc":
        train_data, devloader = load_data_cqcc(batch_size=100, num_train=10000, num_test=10000, balanced=True)
        evalloader = None

    else:

        train_data, devloader, evalloader = load_data(batch_size=100, length=3 * 16000, num_train=10000,
                                                      custom_path=DATA_ROOT, multi_proc=False, balanced=True,
                                                      batch_size_test=100, include_eval=True,
                                                      return_dataset=USE_DATASET,
                                                      short=False)

    dev_eer_list = []
    dev_accuracy_list = []
    eval_eer_list = []
    eval_accuracy_list = []
    for i in range(1):
        if not os.path.isdir("./runs/NEAT"):
            os.makedirs("./runs/NEAT")
        try:
            shutil.rmtree('./runs/NEAT/{}'.format(i))
        except FileNotFoundError:
            pass
        writer = SummaryWriter('./runs/NEAT/{}'.format(i))
        print(i)
        print(dev_eer_list)

        winner, config, stats = run(config_path, 40000)

        eer, accuracy = evaluate_eer_acc(winner, config, devloader,
                                         backprop=backprop, use_gate=USE_GATE, loading_bar=False)
        dev_eer_list.append(eer)
        dev_accuracy_list.append(accuracy)

        eer, accuracy = evaluate_eer_acc(winner, config, evalloader,
                                         backprop=backprop, use_gate=USE_GATE, loading_bar=False)
        eval_eer_list.append(eer)
        eval_accuracy_list.append(accuracy)

        if i == 0:
            make_visualize(winner, config, stats, topology=False)

    print("\n")
    print("DEV equal error rate", dev_eer_list)
    print("accuracy", dev_accuracy_list)

    print("\n")

    array_eer = np.array(dev_eer_list)

    print("DEV EER stats")
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

    print("EVAL EER stats")
    print("min =", array_eer.min())
    print("max =", array_eer.max())
    print("median =", np.median(array_eer))
    print("average =", array_eer.mean())
    print("std =", array_eer.std())
