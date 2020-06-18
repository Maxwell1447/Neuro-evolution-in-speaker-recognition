from __future__ import print_function
import os
import neat
import numpy as np
import random as rd
import multiprocessing

from tqdm import tqdm

from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.data_utils_short import ASVDatasetshort
from anti_spoofing.utils import whiten, gate_activation, evaluate, make_visualize
from anti_spoofing.metrics_utils import rocch2eer, rocch



"""
NEAT APPLIED TO ASVspoof 2019
"""

nb_samples_train = 2538
nb_samples_test = 700

batch_size = 10  # choose an even number

n_processes = 2  # multiprocessing.cpu_count()
n_generation = 5

dev_border = [0, 2548, 6264, 9980, 13696, 17412, 21128, 22296]
index_test = []
index_validation = []
for i in range(len(dev_border)-1):
    index_test += rd.sample([k for k in range(dev_border[i], dev_border[i+1])], 100)

    if i == 0:
        index_validation += rd.sample([k for k in range(dev_border[i], dev_border[i+1])], 300)
    else:
        index_validation += rd.sample([k for k in range(dev_border[i], dev_border[i + 1])], 50)

train_loader = ASVDatasetshort(None, nb_samples=nb_samples_train)
test_loader = ASVDataset(None, is_train=False, is_eval=False, index_list=index_test)
validation_loader = ASVDataset(None, is_train=False, is_eval=False, index_list=index_validation)


class Anti_spoofing_Evaluator(neat.parallel.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, data, config, gc_eval, batch_size=batch_size, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.data = data
        self.config = config
        self.current_batch = []  # contains current batch of audio files
        self.batch_size = batch_size
        self.bona_fide_train = list(range(258))  # index list of bona fide files
        rd.shuffle(self.bona_fide_train)  # shuffle the index
        self.spoofed_train = list(range(258, 2280))  # index list of spoofed files
        rd.shuffle(self.spoofed_train)  # shuffle the index
        self.bona_fide_index = 0
        self.spoofed_index = 0
        self.G = config.pop_size
        self.l_s_n = np.zeros((self.batch_size//2, self.G))
        self.gc = []  # grand champion list
        self.gc_eval = gc_eval

    def evaluate(self, genomes, config):
        jobs = []
        self.next()
        batch_data = self.current_batch
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, self.config, batch_data)))
        
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
            
        champion_indexes = np.argpartition(F, -10)[-10:]
        champions_eer = np.zeros(10)

        index_grand_champion = 0
        for champion_index in champion_indexes:
            genome_id, genome = genomes[champion_index]
            champions_eer[index_grand_champion] = self.gc_eval(genome, self.config, validation_loader)
            index_grand_champion += 1

        grand_champion = genomes[champion_indexes[np.argmax(champions_eer)]]
        self.gc.append(grand_champion)

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


def eer_gc(genome, config, validation_set):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    target_scores = []
    non_target_scores = []
    for data in tqdm(validation_set):
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

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer


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
    # p.add_reporter(neat.Checkpointer(generation_interval=100))

    # Run for up to n_gen generations.
    multi_evaluator = Anti_spoofing_Evaluator(n_processes, eval_genome, train_loader, config_,
                                              gc_eval=eer_gc)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner_))

    # Show output of the most fit genome against training data.
    print('\n')
    winner_net = neat.nn.RecurrentNetwork.create(winner_, config_)

    gc_scores = np.zeros(n_gen)
    for gc_index in range(n_gen):
        _, genome = multi_evaluator.gc[gc_index]
        gc_scores[gc_index] = eer_gc(genome, config_, test_loader)

    _, winner_ = multi_evaluator.gc[np.argmax(gc_scores)]

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


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    winner, config, stats = run(config_path, n_generation)
    make_visualize(winner, config, stats)