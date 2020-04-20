import torch
import os
import neat
from neat.reporting import ReporterSet

import neat_local.visualization.visualize as visualize
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

if __name__ == '__main__':
    config_path = os.path.dirname(__file__) + "/neat.cfg"
    print(config_path)

    random.seed(0)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    genome = config.genome_type(1)
    genome.configure_new(config.genome_config)

    genome.nodes[2] = genome.create_node(config.genome_config, 2)

    genome.connections[0] = genome.create_connection(config.genome_config, -1, 0)
    genome.connections[1] = genome.create_connection(config.genome_config, 0, 0)

    visualize.draw_net(config, genome, True)

