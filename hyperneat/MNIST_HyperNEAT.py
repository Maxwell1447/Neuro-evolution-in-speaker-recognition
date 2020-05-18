from collections.abc import Iterator

import torch
from neat_local.pytorch_neat.cppn import create_cppn
import neat
import neat_local.visualization.visualize as visualize
import os
from hyperneat.modules import LeNet5Cppn
import torchvision

device = torch.device("cuda")


class SampleIterator(Iterator):

    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.i = 0

    def __len__(self):
        return len(self.data) // self.batch_size

    def __next__(self):
        X = torch.empty(self.batch_size, 1, 28, 28)
        y = torch.empty(self.batch_size)
        for b in range(self.batch_size):
            X[b] = self.data[self.i][0]
            y[b] = self.data[self.i][1]
            self.i += 1
            if self.i >= len(self):
                self.i = 0
        return X.to(device), y.to(device)


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

    def __init__(self, eval_function, data_iter, timeout=None):
        self.data_iter = data_iter

    def evaluate(self, genomes, config):
        batch = next(self.data_iter)
        for _, genome in genomes:
            genome.fitness = eval_genome(genome, config, batch)


def load_MNIST(train=True):
    data = torchvision.datasets.MNIST('/TESTS/files/', train=train, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),
                                          torchvision.transforms.Normalize(
                                              (0.1307,), (0.3081,))
                                      ]))

    data_iter = SampleIterator(data, batch_size=50)

    return data_iter


def eval_genome(g, c, batch):
    cppn = create_cppn(
        g, c,
        ['k_x', 'k_y', 'C_in', 'C_out', 'x_in', 'x_out', 'z'],
        ['w', 'b'])

    cnn = LeNet5Cppn(cppn, device).to(device)

    out = cnn(batch[0])

    loss = torch.nn.CrossEntropyLoss()(out, batch[1].long())

    return 1 / (1 + loss.detach().item())


def run(c, num_gen=100):

    pop = neat.Population(c)

    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)

    # evaluator = MNISTParallelEvaluator(multiprocessing.cpu_count(), eval_genome, load_MNIST(train=True))
    winner = pop.run(MNISTEvaluator(eval_genome, load_MNIST(train=True)).evaluate, num_gen)

    visualize.draw_net(c, winner, True, filename="graph_hyperneat_test")
    visualize.plot_stats(stats, ylog=False, view=True, filename="stats_hyperneat_test")
    visualize.plot_species(stats, view=True, filename="species_hyperneat_test1")

    return winner


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)

    config_path = os.path.join(local_dir, 'mnist_neat.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    run(config, num_gen=10000)
