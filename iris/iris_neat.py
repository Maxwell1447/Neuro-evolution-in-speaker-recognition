from __future__ import print_function
import torch
import os
import neat
import neat_local.visualization.visualize as visualize
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"

print(torch.cuda.is_available())

"""
NEAT APPLIED TO IRIS DATASET
"""


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


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 50.
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = np.array(net.activate(xi))
            genome.fitness -= np.linalg.norm(output - xo)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 10000)

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

    plot_decision(winner_net)

    node_names = {-1: labels[0], -2: labels[1], 0: names[0], 1: names[1], 2: names[2]}

    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)


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


if __name__ == '__main__':
    inputs, outputs, labels, names = load_iris([0, 2])

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_iris')

    print(names)
    run(config_path)
