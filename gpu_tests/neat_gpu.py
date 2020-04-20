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
import time

N_HIDDEN = 1732
DROPOUT = 0.


def fully_connect(n_hidden):
    for key_node in range(2, n_hidden + 2):
        genome.nodes[key_node] = genome.create_node(config.genome_config, key_node)

    for key_in in config.genome_config.input_keys + list(genome.nodes.keys()):
        for key_out in genome.nodes:
            key = (key_in, key_out)
            genome.connections[key] = genome.create_connection(config.genome_config, key_in, key_out)


def dropout(_num_drop):
    drop_keys = random.sample(list(genome.connections), _num_drop)
    for key in drop_keys:
        del genome.connections[key]


def test_performance(device="vanilla", input_size=100):
    print()
    inputs = np.zeros(input_size)
    print("TESTING PERFORMANCE")
    print()
    if device == "vanilla":
        t0 = time.time()
        net = neat.nn.RecurrentNetwork.create(genome, config)
        t1 = time.time()
        inputs = inputs.reshape(-1, 1)
        t2 = time.time()
    else:
        t0 = time.time()
        net = rnn.RecurrentNet.create(genome, config, device=device)
        t1 = time.time()
        inputs = torch.from_numpy(inputs)
        inputs.type(torch.float64).to(device).unsqueeze(dim=0)
        t2 = time.time()
    
    print("##### TIMES #####")
    print()
    print("net building time (s): ", t1-t0)
    print("input prepare time (s): ", t2-t1)
    exit(0)
    start = time.time()
    for xi in inputs:
        net.activate(xi)
    end = time.time()
    print("feed time (s): ", end - start)


if __name__ == '__main__':
    config_path = os.path.dirname(__file__) + "/neat.cfg"

    random.seed(0)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # create genome
    genome = config.genome_type(1)
    # initialize the genome with no connection, only input/output nodes
    genome.configure_new(config.genome_config)

    num_hidden = N_HIDDEN
    num_in = len(config.genome_config.input_keys)
    num_node = num_hidden + len(config.genome_config.output_keys)
    dropout_proportion = DROPOUT  # proportion of deleted connections
    num_co = (num_in + num_node) * num_node  # number of connection when fully connected
    num_drop = int(dropout_proportion * num_co)

    fully_connect(num_hidden)
    dropout(num_drop)

    print("number of connections: ", num_co - num_drop)

    # ################### TIME PERFORMANCE ###################### #

    test_performance(device="cpu", input_size=10000)

    # visualize.draw_net(config, genome, True)
