import neat
import matplotlib.pyplot as plt
import torch
from six import itervalues
from tqdm import tqdm

from anti_spoofing.eval_functions import evaluate_eer_acc
from anti_spoofing.metrics_utils import rocch, rocch2eer
from neat_local.nn import RecurrentNet
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils import smooth


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


class EERReporter(neat.reporting.BaseReporter):

    def __init__(self, data, period=50):
        self.data = data
        self.eer = []
        self.acc = []
        self.period = period
        self.gen = 0

    def post_evaluate(self, config, population, species, best_genome):
        if self.gen % self.period == 0:
            with torch.no_grad():
                eer, accuracy = evaluate_eer_acc(best_genome, config, self.data)
            self.eer.append(eer)
            self.acc.append(accuracy)
        self.gen += 1

    def display(self, momentum=0.0):
        plt.figure()
        plt.plot(self.period * np.arange(len(self.eer)),
                 smooth(self.eer, momentum=0.9),
                 label="dev EER every {} generations".format(self.period))
        plt.xlabel("generations")
        plt.ylabel("EER")
        plt.show()

        plt.figure()
        plt.plot(self.period * np.arange(len(self.acc)),
                 smooth(self.acc, momentum=0.9),
                 label="dev acc every {} generations".format(self.period))
        plt.xlabel("generations")
        plt.ylabel("accuracy")
        plt.show()
