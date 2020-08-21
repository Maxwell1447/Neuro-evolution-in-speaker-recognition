from __future__ import division, print_function

import neat
import matplotlib.pyplot as plt
import torch
from neat.reporting import BaseReporter
from six import itervalues
from tqdm import tqdm

from anti_spoofing.eval_functions import evaluate_eer_acc
from anti_spoofing.metrics_utils import rocch, rocch2eer
from neat_local.nn import RecurrentNet
import numpy as np
from torch.utils.data.dataloader import DataLoader
from utils import smooth

import time

from neat.math_util import mean, stdev
from neat.six_util import itervalues, iterkeys


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


class StdOutReporter(BaseReporter):
    """Uses `print` to output information about the run; an example reporter class."""
    def __init__(self, show_species_detail):
        self.show_species_detail = show_species_detail
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0

    def start_generation(self, generation):
        self.generation = generation
        print('\n ****** Running generation {0} ****** \n'.format(generation))
        self.generation_start_time = time.time()

    def end_generation(self, config, population, species_set):
        ng = len(population)
        ns = len(species_set.species)
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            print("   ID   age  size  fitness  adj fit  stag  cplx")
            print("  ====  ===  ====  =======  =======  ====  ====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                c = self.complexity(s)
                f = "--" if s.fitness is None else "{:.4f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >7}  {: >7}  {: >4}  {: >4}".format(sid, a, n, f, af, st, c))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))

    def post_evaluate(self, config, population, species, best_genome):
        # pylint: disable=no-self-use
        fitnesses = [c.fitness for c in itervalues(population)]
        fit_mean = mean(fitnesses)
        fit_std = stdev(fitnesses)
        best_species_id = species.get_species_id(best_genome.key)
        print('Population\'s average fitness: {0:3.5f} stdev: {1:3.5f}'.format(fit_mean, fit_std))
        print(
            'Best fitness: {0:3.5f} - size: {1!r} - species {2} - id {3}'.format(best_genome.fitness,
                                                                                 best_genome.size(),
                                                                                 best_species_id,
                                                                                 best_genome.key))

    def complete_extinction(self):
        self.num_extinctions += 1
        print('All species extinct.')

    def found_solution(self, config, generation, best):
        print('\nBest individual in generation {0} meets fitness threshold - complexity: {1!r}'.format(
            self.generation, best.size()))

    def species_stagnant(self, sid, species):
        if self.show_species_detail:
            print("\nSpecies {0} with {1} members is stagnated: removing it".format(sid, len(species.members)))

    def info(self, msg):
        print(msg)

    @staticmethod
    def complexity(species):
        cplx = []
        for key in species.members:
            cplx.append(len(species.members[key].connections))

        return int(mean(cplx))


class WriterReporter(neat.reporting.BaseReporter):

    def __init__(self, writer, params=None):
        self.writer = writer
        self.params = params if params is not None else []
        self.gen = 0

    def post_evaluate(self, config, population, species, best_genome):
        self.writer.add_scalar("best fitness", best_genome.fitness, self.gen)
        for p in self.params:
            self.writer.add_scalar(p, getattr(config.genome_config, p), self.gen)

        self.gen += 1

    def reset(self):
        self.gen = 0
        self.writer.flush()
