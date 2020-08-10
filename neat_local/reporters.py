import neat
import matplotlib.pyplot as plt
import torch
from six import itervalues
from tqdm import tqdm

from anti_spoofing.metrics_utils import rocch, rocch2eer
from neat_local.nn import RecurrentNet
import numpy as np
from torch.utils.data.dataloader import DataLoader


class ComplexityReporter(neat.reporting.BaseReporter):

    def __init__(self):
        self.conns = []
        self.active_conns = []

    def display(self):
        plt.figure()
        plt.plot(self.conns, label="connections")
        plt.xlabel("generations")
        plt.ylabel("number of connections")
        plt.plot(self.active_conns, label="active_connections")
        plt.show()

    def post_evaluate(self, config, population, species, best_genome):

        co = 0
        a_co = 0
        for key in best_genome.connections:
            co += 1
            if best_genome.connections[key].enabled:
                a_co += 1

        self.conns.append(co)
        self.active_conns.append(a_co)
