from __future__ import print_function
import torch
import os
import neat
import neat_local.visualization.visualize as visualize
from sklearn import datasets

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"

print(torch.cuda.is_available())

"""
NEAT APPLIED TO IRIS DATASET
"""


def load_iris():
    iris = datasets.load_iris()
    iris_x = iris['data']
    iris_y = iris['target']
    # print(iris['feature_names'])
    return iris_x, iris_y


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(inputs, outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo) ** 2


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
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    accuracy = 0
    for xi, xo in zip(inputs, outputs):
        output = winner_net.activate(xi)
        if xo == output:
            accuracy += 1
    accuracy /= outputs.size
    print("**** accuracy = {}  ****".format(accuracy))

    node_names = {-1: 'sepal l', -2: 'sepal w', -3: 'petal l', -4: 'petal w', 0: 'label'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':

    inputs, outputs = load_iris()

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_iris')

    run(config_path)