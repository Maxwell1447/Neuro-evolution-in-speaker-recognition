import os
import neat
import numpy as np
import random as rd
import multiprocessing
from tqdm import tqdm
import pickle

from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.data_utils_short import ASVDatasetshort
from anti_spoofing.utils_ASV import whiten, gate_mfcc, make_visualize
from anti_spoofing.metrics_utils import rocch2eer, rocch

"""
NEAT APPLIED TO ASVspoof 2019
"""

nb_samples_train = 2538  # number of audio files used for training
nb_samples_test = 7000  # number of audio files used for testing

batch_size = 64  # size of the batch used for training, choose a multiple of 12

n_processes = multiprocessing.cpu_count() - 2  # number of workers to use for evaluating the fitness
n_generation = 10  # number of generations

# boundary index of the type of audio files of the dev data set, it will select randomly 100 files from each class
# for testing
dev_border = [0, 2548, 6264, 9980, 13696, 17412, 21128, 22296]
index_test = []
for i in range(len(dev_border) - 1):
    index_test += rd.sample([k for k in range(dev_border[i], dev_border[i + 1])], 1000)


class Anti_spoofing_Evaluator(neat.parallel.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, data, pop, batch_size=batch_size, timeout=None):
        """
        :param num_workers: int
        number of workers to use for evaluating the fitness
        :param eval_function: function
        function to be used to calculate fitness
        :param batch_size: int
        size of the batch used for training, choose an even number
        :param data: ASVDatasetshort
        training data
        :param timeout: int
        how long (in seconds) each subprocess will be given before an exception is raised (unlimited if None).
        """
        super().__init__(num_workers, eval_function, timeout)
        self.data = data
        self.current_batch = []  # contains current batch of audio files
        self.batch_size = batch_size
        self.bona_fide_train = list(range(258))  # index list of bona fide files
        rd.shuffle(self.bona_fide_train)  # shuffle the index
        self.bona_fide_index = 0

        self.train_short_border = [0, 258, 638, 1018, 1398, 1778, 2158, 2538]
        # index list of spoofed files
        self.spoofed_train = np.array([list(range(self.train_short_border[i],
                                                  self.train_short_border[i + 1])) for i in range(1, 7)])
        for sys_id in range(6):
            rd.shuffle(self.spoofed_train[sys_id])  # shuffle the index
        self.spoofed_index = np.array([0, 0, 0, 0, 0, 0])
        self.G = pop
        self.l_s_n = np.zeros((self.batch_size, self.G))

    def evaluate(self, genomes, config):
        """
        Assigns workers to the genomes that will return the false acceptance rate before computing it
        the ease of classification fitness.
        :param genomes: list
        list of all the genomes to get evaluated
        :param config: file
        configuration file
        """
        jobs = []
        self.next()
        batch_data = self.current_batch
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch_data)))

        self.G = len(genomes)
        print(len(batch_data))
        self.l_s_n = np.zeros((len(batch_data), self.G))

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
        """
        change the current_batch attribute of the class to the next batch
        """
        self.current_batch = []

        # adding bona fida index for training
        if batch_size // 2 + self.bona_fide_index >= 259:
            self.bona_fide_index = 0
            rd.shuffle(self.bona_fide_train)
        for index in range(self.batch_size // 2):
            self.current_batch.append(self.data[self.bona_fide_train[self.bona_fide_index + index]])

        # adding spoofed index for training
        for sys_id in range(6):
            if batch_size // 12 + self.spoofed_index[sys_id] >= 381:
                self.spoofed_index[sys_id] = 0
                rd.shuffle(self.spoofed_train[sys_id])
            for index in range(self.batch_size // 12):
                self.current_batch.append(self.data[self.spoofed_train[sys_id, self.spoofed_index[sys_id] + index]])

        self.bona_fide_index += batch_size // 2
        self.spoofed_index += batch_size // 2

        self.current_batch = np.array(self.current_batch)


def eval_genome(genome, config, batch_data):
    """
    Most important part of NEAT since it is here that we adapt NEAT to our problem.
    We tell what is the phenotype of a genome and how to calculate its fitness
    (same idea than a loss)
    :param config: config from the config file
    :param genome: one genome to get evaluated
    :param batch_data: data to use to evaluate the genomes
    :return fitness: returns the fitness of the genome
    this version is intented to use ParallelEvaluator and should be much faster
    """
    net = neat.nn.RecurrentNetwork.create(genome, config)
    target_scores = []
    non_target_scores = []
    l_s_n = np.zeros(len(batch_data))
    for data in batch_data:
        inputs, output = data[0], data[1]
        inputs = whiten(inputs)
        net.reset()
        """
        mask, score = gate_mfcc(net, inputs)
        selected_score = score[mask]
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        """
        xo = gate_mfcc(net, inputs)
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    for i in range(batch_size // 2):
        l_s_n[i] = (non_target_scores >= target_scores[i]).sum() / (batch_size // 2)

    for i in range(non_target_scores.size):
        l_s_n[i + batch_size // 2] = (target_scores <= non_target_scores[i]).sum() / (batch_size // 2)

    return 1 - l_s_n


def run(config_file, n_gen, train_loader):
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
    # p.add_reporter(neat.Checkpointer(generation_interval=40, time_interval_seconds=None))

    # Run for up to n_gen generations.
    multi_evaluator = Anti_spoofing_Evaluator(n_processes, eval_genome, train_loader, pop=config_.pop_size)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner_))

    return winner_, config_, stats_


def evaluate(net, data_loader):
    """
    compute the eer equal error rate
    :param net: network
    :param data_loader: test dataset, contains audio files in a numpy array format
    :return eer
    """
    net.reset()
    target_scores = []
    non_target_scores = []
    for data in tqdm(data_loader):
        sample_input, output = data[0], data[1]
        sample_input = whiten(sample_input)
        xo = gate_mfcc(net, sample_input)
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    train_loader = ASVDatasetshort(None, nb_samples=nb_samples_train, do_mfcc=True)
    test_loader = ASVDataset(None, is_train=False, is_eval=False, index_list=index_test, do_mfcc=True)

    winner, config, stats = run(config_path, n_generation, train_loader)
    make_visualize(winner, config, stats)

    winner_net = neat.nn.RecurrentNetwork.create(winner, config)
    train_eer = evaluate(winner_net, train_loader)
    eer = evaluate(winner_net, test_loader)

    print("\n")
    print("**** training equal error rate = {}  ****".format(train_eer))

    print("\n")
    print("**** equal error rate = {}  ****".format(eer))

    pickle.dump(winner, open('best_genome_eoc_batch_120_c3_balanced_test', 'wb'))
