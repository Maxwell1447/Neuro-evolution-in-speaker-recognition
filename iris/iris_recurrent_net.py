from __future__ import print_function
import torch
import os
import neat
import neat_local.visualization.visualize as visualize
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random

from neat_local.pytorch_neat import t_maze
from neat_local.pytorch_neat.activations import sigmoid_activation
from neat_local.pytorch_neat.recurrent_net import RecurrentNet
from neat_local.pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from neat_local.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from neat_local.pytorch_neat.neat_reporter import LogReporter

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"

print(torch.cuda.is_available())

"""
NEAT APPLIED TO IRIS DATASET
"""


class Evaluator:

    def __init__(self):
        ...

    def eval_genome(self, genome, config, debug=False):
        fitness = 50.
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = torch.tensor(net.activate(xi))
            fitness -= torch.sum((output - xo) ** 2)
        print(fitness)
        return fitness


def activate_net(net, states, debug=False, step_num=0):
    _, out = net.activate(states).max(0)
    print(out)
    return out


def load_iris(pair):
    iris = datasets.load_iris()
    iris_x = iris['data'][:, pair]
    iris_target = iris['target']
    iris_y = []
    for i in range(3):
        iris_y.append((iris_target == i).astype(np.float32))
    iris_y = np.array(iris_y).T

    idx = np.arange(iris_target.size)
    np.random.shuffle(idx)

    labels_ = [iris['feature_names'][i] for i in pair]
    names_ = iris['target_names']
    return iris_x[idx], iris_y[idx], labels_, names_


def run(config_file, n_gen):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    evaluator = Evaluator()

    def eval_genomes(genomes, config):
        for i, (_, genome) in enumerate(genomes):
            try:
                genome.fitness = evaluator.eval_genome(genome, config)
            except Exception as e:
                print(genome)
                raise e

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to n_gen generations.
    winner = p.run(eval_genomes, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\n')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    accuracy = 0
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        if np.argmax(xo) == np.argmax(output):
            accuracy += 1

        if accuracy < 20:
            print(xo, np.argmax(xo))
            print("--> input : ({0:6.4f}, {1:6.4f}) |||  expected : {2}  -  got : {3}".format(xi[0],
                                                                                              xi[1],
                                                                                              names[np.argmax(xo)],
                                                                                              names[np.argmax(output)]
                                                                                              ))
    accuracy *= 3
    accuracy /= outputs.size
    print("**** accuracy = {}  ****".format(accuracy))

    return winner, config, stats


def prediction(clf, xy):
    z = []
    for x, y in xy:
        z.append(np.argmax(clf.activate(np.array([x, y]))))

    return np.array(z)


def plot_decision(clf):
    X = inputs
    y = outputs

    plot_step = 0.02

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = prediction(clf, np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    # Plot the training points
    for i, color in zip(range(3), "ryb"):
        idx = np.where(np.argmax(y, axis=1) == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.savefig("decision_surface.svg")
    plt.show()
    plt.close()


def make_visualize(winner_, config_, stats_):
    winner_net = neat.nn.FeedForwardNetwork.create(winner_, config_)
    plot_decision(winner_net)

    node_names = {-1: labels[0], -2: labels[1], 0: names[0], 1: names[1], 2: names[2]}

    visualize.draw_net(config_, winner_, True, node_names=node_names)
    visualize.plot_stats(stats_, ylog=False, view=True)
    visualize.plot_species(stats_, view=True)


def general_stats(n, config_path):
    max_gen = 1000
    best_seed = (-1, max_gen)
    total_steps = []
    for seed in range(n):
        random.seed(seed)
        winner_, config_, stats_ = run(config_path, max_gen)
        steps = len(stats_.generation_statistics)
        if best_seed[1] > steps:
            best_seed = (seed, steps)
        total_steps.append(steps)

    print("\n best seed = {}   with {} steps".format(best_seed[0], best_seed[1]))

    plt.hist(total_steps, density=True, bins=30)
    plt.xlabel('number of generations')
    plt.savefig("generation histogram.svg")
    plt.show()
    plt.close()


if __name__ == '__main__':
    inputs, outputs, labels, names = load_iris([0, 2])
    data = {"X": inputs, "Y": outputs}

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_iris')

    # general_stats(15, config_path)

    random.seed(7)
    winner, config, stats = run(config_path, 1000)
    make_visualize(winner, config, stats)
