import torch
import os
import neat
from neat.reporting import ReporterSet

import neat_local.visualization.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from neat_local.nn import recurrent_net as rnn
import multiprocessing
import time

N_HIDDEN = 500
DROPOUT = 0.2
INPUT_SIZE = 1000
POPULATION = 100


def fully_connect(genome, n_hidden):
    for key_node in range(2, n_hidden + 2):
        genome.nodes[key_node] = genome.create_node(config.genome_config, key_node)

    for key_in in config.genome_config.input_keys + list(genome.nodes.keys()):
        for key_out in genome.nodes:
            key = (key_in, key_out)
            genome.connections[key] = genome.create_connection(config.genome_config, key_in, key_out)

    return genome


def dropout(genome, num_drop):
    drop_keys = random.sample(list(genome.connections), num_drop)
    for key in drop_keys:
        del genome.connections[key]

    return genome


def eval_genomes(population, conf):
    for (_, g) in tqdm(population):
        eval_genome(g, conf)


def eval_genome(g, conf):
    test_performance(g, conf, input_size=INPUT_SIZE, verbose=False)


def test_performance(g, conf, input_size=100, verbose=True):
    if verbose:
        print()
    inputs = np.zeros(input_size)
    if verbose:
        print("TESTING PERFORMANCE")
    if verbose:
        print()
    if conf.device == "vanilla":
        t0 = time.time()
        net = neat.nn.RecurrentNetwork.create(g, conf)
        t1 = time.time()
        inputs = inputs.reshape(-1, 1)
        t2 = time.time()
    else:
        t0 = time.time()
        net = rnn.RecurrentNet.create(g, conf, device=conf.device)
        t1 = time.time()
        inputs = torch.from_numpy(inputs)
        inputs = inputs.type(torch.float64).to(conf.device).reshape(input_size, 1, 1)
        t2 = time.time()
    if verbose:
        print("##### TIMES #####")
        print()
        print("net building time (s): ", t1-t0)
        print("input prepare time (s): ", t2-t1)
    start = time.time()
    out = []
    for xi in inputs:
        xo = net.activate(xi)
        if conf.device != "vanilla":
            xo = [o.item() for o in xo[0]]
        out.append(xo)
    end = time.time()
    if verbose:
        print("feed time (s): ", end - start)

    return out


def create_genome(conf, key):

    # create genome
    genome = conf.genome_type(key)
    # initialize the genome with no connection, only input/output nodes
    genome.configure_new(conf.genome_config)

    num_hidden = N_HIDDEN
    num_in = len(conf.genome_config.input_keys)
    num_node = num_hidden + len(conf.genome_config.output_keys)
    dropout_proportion = DROPOUT  # proportion of deleted connections
    num_co = (num_in + num_node) * num_node  # number of connection when fully connected
    num_drop = int(dropout_proportion * num_co)

    genome = fully_connect(genome, num_hidden)
    genome = dropout(genome, num_drop)

    return genome


def create_population(conf):
    print("number of connections: ", int((N_HIDDEN + 3) * (N_HIDDEN + 1) * (1 - DROPOUT)))
    population = {}
    for i in range(POPULATION):
        g = create_genome(conf, i)
        population[i] = g

    return list(population.items())


if __name__ == '__main__':
    config_path = os.path.dirname(__file__) + "/neat.cfg"

    random.seed(0)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # ################### TIME PERFORMANCE ###################### #
    config.__setattr__("device", "vanilla")

    pop = create_population(config)
    multi_eval = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    t_0 = time.time()
    multi_eval.evaluate(pop, config)
    t_1 = time.time()
    eval_genomes(pop, config)
    t_2 = time.time()

    print("Multi: ", t_1-t_0)
    print("Simple: ", t_2-t_1)


    # visualize.draw_net(config, genome, True)
