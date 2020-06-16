from __future__ import print_function
import os
import neat
import neat_local.visualization.visualize as visualize
import numpy as np
import random as rd
import multiprocessing

from tqdm import tqdm

from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.data_utils_short import ASVDatasetshort
from anti_spoofing.metrics_utils import rocch2eer, rocch



"""
NEAT APPLIED TO ASVspoof 2019
"""

nb_samples_train = 2538
nb_samples_test = 700

batch_size = 100  # choose an even number

n_processes = multiprocessing.cpu_count()
n_generation = 40

dev_border = [0, 2548, 6264, 9980, 13696, 17412, 21128, 22296]
index_test = []
for i in range(len(dev_border)-1):
    index_test += rd.sample([k for k in range(dev_border[i], dev_border[i+1])], 100)


train_loader = ASVDatasetshort(None, nb_samples=nb_samples_train)
test_loader = ASVDataset(None, is_train=False, is_eval=False, index_list=index_test)


class Anti_spoofing_Evaluator(neat.parallel.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, data, pop, batch_size=batch_size, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.data = data
        self.current_batch = []  # contains current batch of audio files
        self.batch_size = batch_size
        self.bona_fide_train = list(range(258))  # index list of bona fide files
        rd.shuffle(self.bona_fide_train)  # shuffle the index
        self.spoofed_train = list(range(258, 2280))  # index list of spoofed files
        rd.shuffle(self.spoofed_train)  # shuffle the index
        self.bona_fide_index = 0
        self.spoofed_index = 0
        self.G = pop
        self.l_s_n = np.zeros((self.batch_size//2, self.G))


    def evaluate(self, genomes, config):
        jobs = []
        self.next()
        batch_data = self.current_batch
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch_data)))
        
        self.G = len(genomes)
        self.l_s_n = np.zeros((self.batch_size//2, self.G))
        
        
        pseudo_genome_id = 0
        # return ease of classification for each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            self.l_s_n[:, pseudo_genome_id] = job.get(timeout=self.timeout)
            pseudo_genome_id += 1

        # compute the fitness
        p_s = np.sum(self.l_s_n, axis=1).reshape(-1, 1) / self.G
        F = np.sum(self.l_s_n * (1 - p_s), axis=0) / np.sum(1 - p_s)


        pseudo_genome_id = 0
        # assign the fitness back to each genome
        for ignored_genome_id, genome in genomes:
            genome.fitness = F[pseudo_genome_id]
            pseudo_genome_id += 1

    def next(self):
        self.current_batch = []

        # adding bona fida index for training
        if batch_size // 2 + self.bona_fide_index >= 258:
            self.bona_fide_index = 0
            rd.shuffle(self.bona_fide_train)
        for index in range(self.batch_size // 2):
            self.current_batch.append(self.data.__getitem__(self.bona_fide_train[self.bona_fide_index + index]))

        # adding spoofed index for training
        if batch_size // 2 + self.spoofed_index >= 2280:
            self.spoofed_index = 0
            rd.shuffle(self.spoofed_train)
        for index in range(self.batch_size // 2):
            self.current_batch.append(self.data.__getitem__(self.spoofed_train[self.spoofed_index + index]))

        self.bona_fide_index += batch_size // 2
        self.spoofed_index += batch_size // 2

        self.current_batch = np.array(self.current_batch)


def whiten(single_input):
    whiten_input = single_input - single_input.mean()
    var = np.sqrt((whiten_input**2).mean())
    whiten_input *= 1 / var
    return whiten_input


def gate_activation(recurrent_net, inputs):
    length = inputs.size
    score, select = np.zeros(length), np.zeros(length)
    for i in range(length):
        select[i], score[i] = recurrent_net.activate([inputs[i]])
    mask = (select > 0.5)
    return mask, score


def eval_genome(g, config, batch_data):
    net = neat.nn.RecurrentNetwork.create(g, config)
    target_scores = []
    non_target_scores = []
    l_s_n = np.zeros(batch_size // 2)
    for data in batch_data:
        inputs, output = data[0], data[1]
        inputs = whiten(inputs)
        net.reset()
        mask, score = gate_activation(net, inputs)
        selected_score = score[mask]
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    for i in range(batch_size//2):
        l_s_n[i] = (non_target_scores >= target_scores[i]).sum() / (batch_size // 2)

    return 1 - l_s_n


def evaluate(net, data_loader):
    correct = 0
    total = 0
    net.reset()
    target_scores = []
    non_target_scores = []
    for data in tqdm(data_loader):
        inputs, output = data[0], data[1]
        inputs = whiten(inputs)
        mask, score = gate_activation(net, inputs)
        selected_score = score[mask]
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        total += 1
        correct += ((xo > 0.5) == output)
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    rejected_bonafide = (target_scores <= .5).sum()

    return rejected_bonafide, float(correct) / total, eer


def run(config_file, n_gen):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genontype (winner), the configs, the stats of the run and the accuracy on the testing set
    """
    # Load configuration.
    config_ = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config_)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats_ = neat.StatisticsReporter()
    p.add_reporter(stats_)
    p.add_reporter(neat.Checkpointer(generation_interval=40, time_interval_seconds=None))

    # Run for up to n_gen generations.
    multi_evaluator = Anti_spoofing_Evaluator(n_processes, eval_genome, train_loader, pop=config_.pop_size)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner_))

    # Show output of the most fit genome against training data.
    print('\n')
    winner_net = neat.nn.RecurrentNetwork.create(winner_, config_)

    train_bonafide_rejected, train_accuracy, train_eer = evaluate(winner_net, train_loader)
    bonafide_rejected, accuracy, eer = evaluate(winner_net, test_loader)

    print("\n")
    print("**** accuracy = {}  ****".format(train_accuracy))
    print("**** number of bone fide rejected = {}  ****".format(train_bonafide_rejected))
    print("**** equal error rate = {}  ****".format(train_eer))

    print("\n")
    print("**** accuracy = {}  ****".format(accuracy))
    print("**** number of bone fide rejected = {}  ****".format(bonafide_rejected))
    print("**** equal error rate = {}  ****".format(eer))

    return winner_, config_, stats_


def make_visualize(winner_, config_, stats_):
    """
    Plot and draw:
        - the graph of the topology
        - the fitness evolution over generations
        - the speciation evolution over generations
    :param winner_:
    :param config_:
    :param stats_:
    :return:
    """
    winner_net = neat.nn.FeedForwardNetwork.create(winner_, config_)

    node_names = {-1: "input", 1: "score", 0: "gate"}

    visualize.plot_stats(stats_, ylog=False, view=True)
    visualize.plot_species(stats_, view=True)
    visualize.draw_net(config_, winner_, True, node_names=node_names)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    winner, config, stats = run(config_path, n_generation)
    make_visualize(winner, config, stats)