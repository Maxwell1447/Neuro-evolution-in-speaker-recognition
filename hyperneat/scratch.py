import multiprocessing

import torch
from neat_local.pytorch_neat.cppn import create_cppn
import numpy as np
import neat
import neat_local.visualization.visualize as visualize
import os
import soundfile as sf
from torch.nn import BCELoss
from neat.parallel import ParallelEvaluator
from hyperneat.modules import AudioCNN
from hyperneat.TEST import test


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
    return torch.from_numpy(np.array(data)).float()


def eval_genome(g, c):

    cppn = create_cppn(
        g, c,
        ['k', 'C_in', 'C_out', 'z'],
        ['w', 'b'])

    cnn = AudioCNN(cppn)

    criterion = BCELoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001)
    y_ = getattr(c, "y")

    loss = None

    for i in range(20):

        out = cnn(c.data.view(c.data.shape[0], 1, -1))
        optimizer.zero_grad()
        out = torch.sigmoid(cnn.l1(out.view(-1, 72)))
        loss = criterion(out, y_)
        loss.backward()
        optimizer.step()

    return 1 / (1 + loss.detach().item())


def eval_genomes(genomes, c):
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, c)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)

    config_path = os.path.join(local_dir, 'neat.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)


    # xx, yy = torch.meshgrid([torch.arange(50).float(), torch.arange(50).float()])
    #
    # dd = ((xx - 14)**2 + (yy - 14)**2).sqrt()
    #
    # plt.imshow(torch.tanh(node(x=xx, y=yy, d=dd)), cmap='Greys')
    # plt.show()
    # plt.imshow(torch.tanh(cppn[1](x=xx, y=yy, d=dd)), cmap='Greys')
    # plt.show()



    data = load_audio()
    y = torch.tensor([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]).view(-1, 1)
    setattr(config, "y", y)
    setattr(config, "data", data)

    pop = neat.Population(config)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    winner = pop.run(eval_genomes, 200)

    visualize.draw_net(config, winner, True, filename="graph_hyperneat_test")
    visualize.plot_stats(stats, ylog=False, view=True, filename="stats_hyperneat_test")
    visualize.plot_species(stats, view=True, filename="species_hyperneat_test1")
