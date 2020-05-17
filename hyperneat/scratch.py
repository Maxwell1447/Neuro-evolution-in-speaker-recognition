import multiprocessing

import torch

from gpu_tests.neat_gpu import fully_connect, dropout
from neat_local.pytorch_neat.cppn import create_cppn
import numpy as np
import neat
import neat_local.visualization.visualize as visualize
from neat_local.pytorch_neat.adaptive_net import AdaptiveNet
from neat_local.pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
import os
import soundfile as sf
from torch.nn import BCELoss
from neat.parallel import ParallelEvaluator
from neat_local.pytorch_neat.activations import tanh_activation
from neat_local.pytorch_neat.neat_reporter import LogReporter
import matplotlib.pyplot as plt


def create_genome(conf, key):
    # create genome
    genome = conf.genome_type(key)
    # initialize the genome with no connection, only input/output nodes
    genome.configure_new(conf.genome_config)

    # num_hidden = 5
    # num_in = len(conf.genome_config.input_keys)
    # num_node = num_hidden + len(conf.genome_config.output_keys)
    # dropout_proportion = 0.5  # proportion of deleted connections
    # num_co = (num_in + num_node) * num_node  # number of connection when fully connected
    # num_drop = int(dropout_proportion * len(list(genome.connections)))
    #
    # genome = fully_connect(genome, num_hidden, conf, recurrent=False)
    # genome = dropout(genome, num_drop)

    return genome


def load_audio(name="LA_T_1000137.flac"):
    path = "E:\\Eurecom\\Project\\git_deposit\\anti_spoofing\\data\\LA\\ASVspoof2019_LA_train\\flac\\"

    # print(os.listdir(path))
    data = []
    for file in os.listdir(path)[:10]:
        audio, sample_rate = sf.read(path + file)
        length = audio.size
        audio = np.tile(audio, int(np.floor(48000 / length)) + 1)
        data.append(audio[:48000])
    return torch.from_numpy(np.array(data))


def eval_genome(g, c):
    # net = AdaptiveNet.create(g, c,
    #                          np.array([[x, 0.] for x in np.linspace(-10, 10, 480)]),
    #                          np.array([[x, 0.5] for x in np.linspace(-1, 1, 100)]),
    #                          [[-0.5, 1.], [0.5, 1.]],
    #                          device="cpu",
    #                          activation=tanh_activation,
    #                          batch_size=10)
    net = AdaptiveLinearNet.create(g, c,
                                   np.array([[x, 0.] for x in np.linspace(-15, 15, 480)]),
                                   [[-0.5, 1.], [0.5, 1.]],
                                   device="cpu",
                                   activation=tanh_activation,
                                   batch_size=10)
    mask = torch.empty(10, 100)
    score = torch.empty(10, 100)
    data_ = getattr(c, "data")
    for i, inputs in enumerate(data_.reshape(100, 10, -1)):
        out = torch.sigmoid(net.activate(inputs))
        mask[:, i] = out[:, 0]
        score[:, i] = out[:, 1]

    pred = (mask * score).sum(dim=1) / mask.sum(dim=1)

    loss = BCELoss()
    y_ = getattr(c, "y")
    return 1 / (1 + loss(y_, pred).detach().item())


def eval_genomes(genomes, c):
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, c)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)

    config_path = os.path.join(local_dir, 'neat.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    genome = create_genome(config, 0)

    cppn = create_cppn(
        genome, config,
        ['x', 'y', 'd'],
        ['out_pattern'])

    node = cppn[0]
    inputs = torch.zeros(50, 50)

    xx, yy = torch.meshgrid([torch.arange(50).float(), torch.arange(50).float()])

    dd = ((xx - 14)**2 + (yy - 14)**2).sqrt()

    plt.imshow(torch.tanh(node(x=xx, y=yy, d=dd)), cmap='Greys')
    plt.show()
    plt.imshow(torch.tanh(cppn[1](x=xx, y=yy, d=dd)), cmap='Greys')
    plt.show()

    # data = load_audio()
    # y = torch.tensor([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.])
    # setattr(config, "y", y)
    # setattr(config, "data", data)
    #
    # evaluator = ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    #
    # pop = neat.Population(config)
    #
    # stats = neat.StatisticsReporter()
    # pop.add_reporter(stats)
    # reporter = neat.StdOutReporter(True)
    # pop.add_reporter(reporter)
    #
    # winner = pop.run(eval_genomes, 300)
    #
    # visualize.draw_net(config, winner, True, filename="graph_hyperneat_test")
    # visualize.plot_stats(stats, ylog=False, view=True, filename="stats_hyperneat_test")
    # visualize.plot_species(stats, view=True, filename="species_hyperneat_test1")
