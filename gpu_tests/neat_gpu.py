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
from itertools import product
import pandas as pd

N_HIDDEN = 50
DROPOUT = 0.2
INPUT_SIZE = 1000
POPULATION = 100


def fully_connect(genome, n_hidden, config, recurrent=True):
    """
    Fully connects the nodes of a genome, allowing recurrency.
    
    I = Input nodes
    H = Hidden nodes
    O = Output nodes
    
    I -> H
    I -> O
    H -> H
    H -> O
    O -> H
    O -> O
    """
    
    # Create hidden nodes
    for key_node in range(2, n_hidden + 2):
        genome.nodes[key_node] = genome.create_node(config.genome_config, key_node)

    if recurrent:
        nodes = config.genome_config.input_keys + list(genome.nodes.keys())
    else:
        nodes = config.genome_config.input_keys
    for key_in in nodes:
        for key_out in genome.nodes:
            key = (key_in, key_out)
            genome.connections[key] = genome.create_connection(config.genome_config, key_in, key_out)

    return genome


def dropout(genome, num_drop):
    """
    Drops num_drop connection (to create sparse connections)
    """
    
    drop_keys = random.sample(list(genome.connections), num_drop)
    for key in drop_keys:
        del genome.connections[key]

    return genome


def eval_genomes(population, conf):
    """
    Evaluate all the genomes of the population.
    Note: The evaluation just feeds the nets with the input sequence, there is no fitness calculation here.
    """
    for (_, g) in population:
        eval_genome(g, conf)


def eval_genome(g, conf):
    """
    Evaluate a single genome (just feed, no fitness calculation)
    """
    test_performance(g, conf, input_size=None, verbose=False)


def test_performance(g, conf, input_size=None, verbose=True):
    """
    Feed a phenotype describe by conf.device
    conf.device can be "vanilla", "cpu" or "cuda"
    """
    if input_size is None:
        input_size = conf.input_size
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
        print("net building time (s): ", t1 - t0)
        print("input prepare time (s): ", t2 - t1)
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
    """
    :param conf: config parameter
    :param key: key uniquely identifying the genome
    :return: the built genome according to parameters N_HIDDEN and DROPOUT
    """
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


def create_population(conf, verbose=False):
    """
    Creates a population of genomes.
    """
    if verbose:
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

    multi_eval = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

    N = [10, 20, 50, 80, 100]
    D = [0.0, 0.5, 0.8]
    I = [100, 500, 1000, 2000]
    P = [20, 50, 100]
    devices = ["vanilla", "cpu", "cuda"]
    df = pd.DataFrame(columns=["N", "D", "I", "P", "device", "M", "S"])
    df.to_csv("time_stats_local.csv")
    prod = product(N, D, I, P, devices) # list of every possible combination (cartesian product)
    for (n, d, i, p, device) in tqdm(prod, total=len(N)*len(D)*len(I)*len(P)*len(devices)):

        N_HIDDEN = n
        DROPOUT = d
        INPUT_SIZE = i
        POPULATION = p
        config.__setattr__("device", device)
        config.__setattr__("input_size", i)
        pop = create_population(config, verbose=False)
        multi_eval = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)

        t_0 = time.time()
        if device != "cuda":
            multi_eval.evaluate(pop, config)
        t_1 = time.time()
        eval_genomes(pop, config)
        t_2 = time.time()

        dt = t_1 - t_0 if device != "cuda" else None
        df = pd.DataFrame(columns=["N", "D", "I", "P", "device", "M", "S"])
        df = df.append(pd.Series([n, d, i, p, device, dt, t_2 - t_1], index=df.columns), ignore_index=True)
        df.to_csv("time_stats_local.csv", mode='a', header=False)

    print(df.head())

