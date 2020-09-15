import os
import neat
import backprop_neat
import numpy as np
import multiprocessing
import warnings

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from anti_spoofing.utils_ASV import make_visualize, show_stats
from anti_spoofing.data_loader import load_data
from anti_spoofing.eval_funtion_ce import eval_genome_ce, ProcessedASVEvaluator, feed_and_predict_ce
from anti_spoofing.metrics_utils import rocch2eer, rocch

warnings.filterwarnings("ignore", category=UserWarning)

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
    stats_ = backprop_neat.StatisticsReporter()
    p.add_reporter(stats_)
    p.add_reporter(backprop_neat.Checkpointer(generation_interval=1000, time_interval_seconds=None))

    # Run for up to n_gen generations.
    multi_evaluator = ProcessedASVEvaluator(multiprocessing.cpu_count(), eval_genome_ce, trainloader)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # Display the winning genome.
    # print('\nBest genome:\n{!s}'.format(winner_))

    return winner_, config_, stats_


def evaluate(g, conf, data):
    """
    returns the equal error rate, the accuracy (both multi class and binary class) and the confusion matrix
    """
    predictions, targets = feed_and_predict_ce(data, g, conf)
    print(predictions)
    target_scores = predictions[targets == 0]
    non_target_scores = predictions[targets > 0]
    pmiss, pfa = rocch(target_scores[:, 0], non_target_scores[:, 0])
    eer = rocch2eer(pmiss, pfa)

    predictions = np.argmax(predictions, axis=1)
    accuracy = (predictions == targets).sum() / len(data)


    c_matrix = confusion_matrix(targets, predictions, normalize="pred")

    predictions[predictions > 0] = 1
    targets[targets > 0] = 1

    anti_spoofing_accuracy = (predictions == targets).sum() / len(data)

    return eer, accuracy, anti_spoofing_accuracy, c_matrix


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed_ce.cfg')

    trainloader, testloader, evalloader = load_data(batch_size=100, length=3 * 16000, num_train=30000, num_test=40000,
                                                    metadata=True, batch_size_test=100, option="lfcc", multi_proc=True,
                                                    include_eval=True, balanced=False)


    winner, config, stats = run(config_path, 1000)

    eer, accuracy, anti_spoofing_accuracy, c_matrix = evaluate(winner, config, testloader)
    eer_eval, accuracy_eval, anti_spoofing_accuracy_eval, c_matrix_eval = evaluate(winner, config, evalloader)

    print("\n")
    print("Result on dev")
    print("equal error rate", eer)
    print("accuracy", accuracy)
    print("accuracy", anti_spoofing_accuracy)

    print("\n")
    print("Result on eval")
    print("equal error rate", eer_eval)
    print("accuracy", accuracy_eval)
    print("accuracy", anti_spoofing_accuracy_eval)



    make_visualize(winner, config, stats)

    plt.figure(figsize=(14, 7))
    yticklabels = ["bona fide", "Wavenet vocoder", "Conventional vocoder WORLD",
                   "Conventional vocoder MERLIN", "Unit selection system MaryTTS",
                   "Voice conversion using neural networks", "transform function-based voice conversion"]
    plt.title("Confusion matrix bonafide, spoofed", fontsize=16)
    sns.heatmap(c_matrix, annot=True, linewidths=.5, cmap="coolwarm", fmt=".2%", yticklabels=yticklabels)
    plt.show()


    plt.figure(figsize=(14, 7))
    yticklabels = ["bona fide", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08",
                   "A09", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17",
                   "A18", "A19"]
    plt.title("Confusion matrix bonafide, spoofed", fontsize=16)
    sns.heatmap(c_matrix_eval, annot=True, linewidths=.5, cmap="coolwarm", fmt=".2%", yticklabels=yticklabels)
    plt.show()

