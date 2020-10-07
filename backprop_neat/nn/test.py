import os

import neat
import random
import torch
from backprop_neat.nn.feed_forward import FeedForwardNetwork

from neat.nn.feed_forward import FeedForwardNetwork as VanillaFeedForwardNetwork

N_HIDDEN = 20
DROPOUT = 0.5


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

    if not recurrent:
        for key_in in genome.nodes:
            for key_out in config.genome_config.output_keys:
                if key_in not in config.genome_config.output_keys:
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
    num_out = len(conf.genome_config.output_keys)
    dropout_proportion = DROPOUT  # proportion of deleted connections
    # number of connection when fully connected
    num_co = (num_in + num_hidden) * num_out + num_in * (num_hidden + num_out)
    num_drop = int(dropout_proportion * num_co)

    genome = fully_connect(genome, num_hidden, conf, recurrent=False)
    genome = dropout(genome, num_drop)

    return genome


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    genome = create_genome(config, 0)

    net = FeedForwardNetwork.create(genome, config)

    input = torch.ones((10, 20), dtype=net.dtype, device=net.device)

    print(net.activate(input))

    vanilla_net = VanillaFeedForwardNetwork.create(genome, config)

    print(len(vanilla_net.input_nodes))

    input = 20 * [1]

    print(vanilla_net.activate(input))
