from __future__ import print_function
import torch
import os
import neat
import neat_local.nn.feed_forward as feed_forward
import neat_local.visualization.visualize as visualize
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"

print(torch.cuda.is_available())

"""
NEAT APPLIED TO IRIS DATASET
"""


def load_iris(features, wrong_labelling=0):
    """
    :param features: indices we choose to keep
    :param wrong_labelling: number of point randomly labeled
    :return: the data train-test split, the inputs and outputs string names (eg. "sepal length (cm)", "versicolor"
    """
    iris = datasets.load_iris()
    iris_x = iris['data'][:, features]
    iris_target = iris['target']

    X_train, X_test, y_train, y_test = train_test_split(iris_x, iris_target, test_size=0.33, random_state=42)

    # wrong labelling
    perturbation_idx = np.random.choice(y_train.shape[0], wrong_labelling)
    y_train[perturbation_idx] = np.random.randint(3, size=wrong_labelling)

    y_train_reformated = []
    for i in range(3):
        y_train_reformated.append((y_train == i).astype(np.float32))
    y_train_reformated = np.array(y_train_reformated).T

    y_test_reformated = []
    for i in range(3):
        y_test_reformated.append((y_test == i).astype(np.float32))
    y_test_reformated = np.array(y_test_reformated).T



    labels_ = [iris['feature_names'][i] for i in features]
    names_ = iris['target_names']

    data = {"X_train": X_train,
            "X_test": X_test,
            "y_train": y_train_reformated,
            "y_test": y_test_reformated}
    return data, labels_, names_


def eval_genomes(genomes, config):
    """
    Most important part of NEAT since it is here that we adapt NEAT to our problem.
    We tell what is the phenotype of a genome and how to calculate its fitness (same idea than a loss)

    :param genomes: list of all the genomes to get evaluated
    :param config: config from the config file
    """
    for genome_id, genome in genomes:
        genome.fitness = 30.
        net = feed_forward.FeedForwardNetwork.create(genome, config)
        # net = RecurrentNet.create(genome, config, device=device)
        for xi, xo in zip(inputs, outputs):
            output = np.array(net.activate(xi))
            # update the fitness of each individual:
            # fitness = 30 - MSE_all_data
            genome.fitness -= np.sum((output - xo) ** 2)


def run(config_file, n_gen):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genontype (winner), the configs, the stats of the run and the accuracy on the testing set
    """
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

    # Run for up to n_gen generations.
    winner = p.run(eval_genomes, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\n')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    accuracy = 0
    for xi, xo in zip(data["X_test"], data["y_test"]):
        output = winner_net.activate(xi)
        if np.argmax(xo) == np.argmax(output):
            accuracy += 1

    accuracy *= 3
    accuracy /= data["y_test"].size
    print("**** accuracy = {}  ****".format(accuracy))

    return winner, config, stats, accuracy


def prediction(clf, xy):
    """
    Useful for the decision surface plot
    :param clf: our phenotype
    :param xy: meshgrid surface
    :return: the decision for each point on the surface
    """
    z = []
    for x, y in xy:
        z.append(np.argmax(clf.activate(np.array([x, y]))))

    return np.array(z)


def plot_decision(clf):
    """
    Plot the decision grid of clf
    :param clf: a given phenotype
    """
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


def make_visualize(winner_, config_, stats_, decision_surface=True):
    """
    Plot and draw:
        - the graph of the topology
        - the decision surface
        - the fitness evolution over generations
        - the speciation evolution over generations
    :param winner_:
    :param config_:
    :param stats_:
    :param decision_surface:
    :return:
    """
    winner_net = neat.nn.FeedForwardNetwork.create(winner_, config_)
    if decision_surface:
        plot_decision(winner_net)

    node_names = {0: names[0], 1: names[1], 2: names[2]}
    for i in range(len(features)):
        node_names[-1-i] = labels[i]

    visualize.draw_net(config_, winner_, True, node_names=node_names)
    visualize.plot_stats(stats_, ylog=False, view=True)
    visualize.plot_species(stats_, view=True)


def general_stats(n, config_path):
    """
    Launch runs to get stats on them and show the distribution of the number of generation and the accuracy
    :param n: number of runs
    :param config_path: path to the config file
    """

    max_gen = 1000              # max number of generation before stopping
    best_seed = (-1, max_gen)
    total_steps = []
    accuracies = []
    for seed in range(n):
        random.seed(seed)
        winner_, config_, stats_, acc = run(config_path, max_gen)
        accuracies.append(acc)
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

    plt.hist(accuracies, density=True, bins=30)
    plt.xlabel('accuracy on test set')
    plt.savefig("accuracy repartition.svg")
    plt.show()
    plt.close()

    print(np.mean(np.array(accuracies)))


if __name__ == '__main__':

    features = [0, 1, 2, 3]         # features indices we choose to keep (subset of [0, 1, 2, 3])
    data, labels, names = load_iris(features, wrong_labelling=0)
    inputs, outputs = data["X_train"], data['y_train']
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_iris')

    # for general stats over several runs
    # general_stats(25, config_path)

    # for the result of just one run
    random.seed(0)
    winner, config, stats, acc = run(config_path, 100)
    make_visualize(winner, config, stats, decision_surface=(len(features) == 2))
