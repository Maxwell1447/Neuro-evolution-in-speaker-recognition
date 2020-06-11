from collections.abc import Iterator

import torch
from neat_local.pytorch_neat.cppn import create_cppn
import neat
import neat_local.visualization.visualize as visualize
import os
from hyperneat.modules import LeNet5Cppn, LeNet5MSS, MNISTCNNClassical
import torchvision
import matplotlib.pyplot as plt
from utils import smooth
import time
import numpy as np
import optuna
import pandas as pd

device = torch.device("cuda")


class SampleIterator(Iterator):

    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.i = 0
        self.shuffle = np.random.permutation(len(self.data))
        self.shuffle = np.arange(len(self.data))

    def __len__(self):
        return len(self.data) // self.batch_size

    def __next__(self):
        self.i = 0
        X = torch.empty(self.batch_size, 1, 28, 28)
        y = torch.empty(self.batch_size)
        for b in range(self.batch_size):
            X[b] = self.data[self.shuffle[self.i]][0]
            y[b] = self.data[self.shuffle[self.i]][1]
            self.i += 1
            if self.i >= len(self):
                self.i = 0
                self.shuffle = np.random.permutation(len(self.data))
                # self.shuffle = np.arange(len(self.data))
        return X.to(device), y.to(device).long()


class MNISTParallelEvaluator(neat.ParallelEvaluator):

    def __init__(self, num_workers, eval_function, data_iter, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.data_iter = data_iter

    def evaluate(self, genomes, config):
        batch = next(self.data_iter)
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)


class MNISTEvaluator:

    def __init__(self, eval_function, data_iter):
        self.data_iter = data_iter
        self.eval_function = eval_function

    def evaluate(self, genomes, config):
        batch = next(self.data_iter)
        for _, genome in genomes:
            a = time.time()
            genome.fitness = 100 * self.eval_function(genome, config, batch, acc=False)
            b = time.time()

    def accuracy(self, genome, config):
        batch = next(self.data_iter)
        return self.eval_function(genome, config, batch, acc=True)


class AccuracyReporter(neat.reporting.BaseReporter):

    def __init__(self, evaluator):
        self.accuracy = []
        self.evaluator = evaluator

    def post_evaluate(self, config, population, species, best_genome):
        self.accuracy.append(self.evaluator.accuracy(best_genome, config))


def load_MNIST(train=True, batch_size=10):
    data = torchvision.datasets.MNIST('/TESTS/files/', train=train, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))

    data_iter = SampleIterator(data, batch_size=batch_size)

    return data_iter


def eval_genome(g, c, batch, acc=False):
    # cppn = create_cppn(
    #     g, c,
    #     ['k_x', 'k_y', 'x_out'],
    #     [])
    cppn = create_cppn(
        g, c,
        ['k_x', 'k_y', 'C_in', 'C_out', 'x_in', 'x_out', 'z'],
        [])

    cnn = MNISTCNNClassical(cppn, device).to(device)

    # print("offset =", cnn.l1.offset)zqsdqszdqqzzzzz

    out = cnn(batch[0])

    if acc:
        # torch.nn.functional.one_hot(out, num_classes=10)
        y_pred = out.argmax(dim=1)
        print("true: ", batch[1].long())
        print("pred: ", y_pred)
        return ((y_pred == batch[1].long()).float().sum() / len(y_pred)).item()

    loss = torch.nn.CrossEntropyLoss()(out, batch[1].long())

    return 1 / (1 + loss.detach().item())


def run(c, num_gen=100, params=None, display=False):

    if params is None:
        params = {}
    for p in params:
        setattr(c.genome_config, p, params[p])


    pop = neat.Population(c)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    evaluator = MNISTEvaluator(eval_genome, load_MNIST(train=True))
    # evaluator = MNISTParallelEvaluator(multiprocessing.cpu_count(), eval_genome, load_MNIST(train=True))

    acc_reporter = AccuracyReporter(evaluator)
    pop.add_reporter(acc_reporter)

    start = time.time()
    winner = pop.run(evaluator.evaluate, num_gen)
    end = time.time()

    print("execution time:", end - start)

    if display:
        visualize.draw_net(c, winner, True, filename="graph_hyperneat_test")
        visualize.plot_stats(stats, ylog=False, view=True, filename="stats_hyperneat_test")
        plt.figure()
        plt.plot(smooth(acc_reporter.accuracy, momentum=0.995))
        plt.show()
        visualize.plot_species(stats, view=True, filename="species_hyperneat_test1")

    return winner


def objective(trial):

    local_dir = os.path.dirname(__file__)

    config_path = os.path.join(local_dir, 'mnist_neat.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    conn_add_prob = trial.suggest_uniform("conn_add_prob", 0.01, 0.2)  # 0.01, 0.2
    # conn_delete_prob = trial.suggest_uniform("conn_delete_prob", 0.005, 0.1)  # 0.005, 0.1
    #
    # node_add_prob = trial.suggest_uniform("node_add_prob", 0.001, 0.2)  # 0.001, 0.2
    # node_delete_prob = trial.suggest_uniform("node_delete_prob", 0.001, 0.1)  # 0.001, 0.1
    #
    # bias_mutate_power = trial.suggest_uniform("bias_mutate_power", 0.01, 1.)  # 0.01, 1.
    # weight_mutate_power = trial.suggest_uniform("weight_mutate_power", 0.01, 1.)  # 0.01, 1.
    #
    # compatibility_disjoint_coefficient = trial.suggest_uniform("compatibility_disjoint_coefficient", 0.8, 1.2)  # 0.8, 1.2
    # compatibility_weight_coefficient = trial.suggest_uniform("compatibility_weight_coefficient", 0.3, 0.7)  # 0.3, 0.7

    conn_add_prob = 0.01
    conn_delete_prob = 0.03
    node_add_prob = 0.066
    node_delete_prob = 0.05
    bias_mutate_power = 0.6
    weight_mutate_power = 0.5
    compatibility_disjoint_coefficient = 1.15
    compatibility_weight_coefficient = 0.5

    params = {"conn_add_prob": conn_add_prob,
              "conn_delete_prob": conn_delete_prob,
              "node_add_prob": node_add_prob,
              "node_delete_prob": node_delete_prob,
              "bias_mutate_power": bias_mutate_power,
              "weight_mutate_power": weight_mutate_power,
              "compatibility_disjoint_coefficient": compatibility_disjoint_coefficient,
              "compatibility_weight_coefficient": compatibility_weight_coefficient
              }

    winner = run(config, num_gen=2500, params=params)

    data_iter = load_MNIST(train=True, batch_size=10)

    batch = next(data_iter)
    fitness = eval_genome(winner, config, batch, acc=False)

    return fitness


if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    df = study.trials_dataframe()

    df.to_csv("best_params_runs.csv", index=False)