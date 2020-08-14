import torch

import backprop_neat
import os
import random as rd
from backprop_neat.six_util import iteritems, itervalues


def test_genome_creation(config):
    g = config.genome_type(0)
    # initialize the genome with no connection, only input/output nodes
    g.configure_new(config.genome_config)

    return g


def check_params(genome):
    params = genome.get_params()
    for p in params:
        assert isinstance(p, torch.Tensor) and p.requires_grad


def test_pop_creation(config):
    p = backprop_neat.Population(config)

    return p


def attribute_fitness(pop):
    for g in itervalues(pop.population):
        g.fitness = rd.random()


def test_reproduce(pop):
    population = pop.reproduction.reproduce(pop.config, pop.species,
                                            pop.config.pop_size, pop.generation)

    for g in itervalues(population):
        check_params(g)


def test_activate(g):
    x = torch.randn(1, 20)
    
    net = backprop_neat.nn.RecurrentNet.create(g, config, dtype=torch.float32)

    print(net.input_to_output)
    
    y = net.activate(x)

    print(y.shape, y)


def test_backprop(g):

    params = g.get_params()
    optimizer = torch.optim.Adam(params, lr=0.1)
    optimizer.zero_grad()

    x = torch.randn(1, 20)

    net = backprop_neat.nn.RecurrentNet.create(g, config, dtype=torch.float32)

    y = net.activate(x)

    y.sum().backward()
    for i in range(10):
        optimizer.step()


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "neat_test.cfg")

    config = backprop_neat.Config(backprop_neat.DefaultGenome, backprop_neat.DefaultReproduction,
                                  backprop_neat.DefaultSpeciesSet, backprop_neat.DefaultStagnation,
                                  config_path)

    g = test_genome_creation(config)

    check_params(g)

    p = test_pop_creation(config)

    attribute_fitness(p)
    test_reproduce(p)
    
    # test_activate(g)
    
    test_backprop(g)