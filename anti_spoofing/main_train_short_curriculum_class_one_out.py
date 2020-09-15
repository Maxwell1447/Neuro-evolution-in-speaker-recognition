import os
import neat
import numpy as np
import random as rd
import multiprocessing
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import neat_local.visualization.visualize as visualize
from anti_spoofing.data_utils import ASVDataset, ASVFile
from anti_spoofing.data_utils_short import ASVDatasetshort
from anti_spoofing.utils_ASV import gate_lfcc, show_stats, show_stats_no_percentage
from anti_spoofing.metrics_utils import rocch2eer, rocch

"""
NEAT APPLIED TO ASVspoof 2019
"""

nb_samples_train = 2538  # number of audio files used for training
nb_samples_test = 7000  # number of audio files used for testing

batch_size = 124  # size of the batch used for training, choose an even number

n_processes = multiprocessing.cpu_count() - 2  # number of workers to use for evaluating the fitness
n_generation = 150  # number of generations
n_iterations = 20  # number of iterations

# boundary index of the type of audio files of the train short data set for testing++
train_short_border = [0, 258, 638, 1018, 1398, 1778, 2158, 2538]

# class used for validation
class_out = 6
index_validation = list(range(258)) + list(range(train_short_border[class_out], train_short_border[class_out + 1]))


class Anti_spoofing_Evaluator(neat.parallel.ParallelEvaluator):
    def __init__(self, num_workers, eval_function, data, validation_data, pop_size, batch_size=batch_size,
                 timeout=None):
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
        self.train_border = train_short_border
        self.bona_fide_train = list(range(258))  # index list of bona fide files
        self.class_out = class_out
        rd.shuffle(self.bona_fide_train)  # shuffle the index
        if self.class_out == 1:
            self.spoofed_train = list(range(self.train_border[2], self.train_border[3]))
        elif self.class_out == 2:
            self.spoofed_train = list(range(258, self.train_border[2]))
        self.spoofed_train = list(range(258, self.train_border[3]))  # index list of spoofed files
        rd.shuffle(self.spoofed_train)  # shuffle the index
        self.bona_fide_index = 0
        self.spoofed_index = 0
        self.G = pop_size
        self.l_s_n = np.zeros((self.batch_size, self.G))
        self.nb_generations = 0
        self.validation_data = validation_data
        self.validation_eer = np.zeros((n_generation, 10))
        self.best_eer_validation = 0.5
        self.gc = None
        self.app_gc = 0

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

        if self.nb_generations == 50:
            if self.class_out == 4:
                self.spoofed_train += list(range(self.train_border[4], self.train_border[5]))
            elif self.class_out == 5:
                self.spoofed_train += list(range(self.train_border[5], self.train_border[6]))
            else:
                self.spoofed_train += list(range(self.train_border[4], self.train_border[6]))
            self.spoofed_index = 2500
            print("**********  Added class 4 and 5 for training *********")
        if self.nb_generations == 100:
            if self.class_out == 3:
                self.spoofed_train += list(range(self.train_border[6], self.train_border[7]))
            elif self.class_out == 6:
                self.spoofed_train += list(range(self.train_border[3], self.train_border[4]))
            else:
                self.spoofed_train += list(range(self.train_border[3], self.train_border[4]))
                self.spoofed_train += list(range(self.train_border[6], self.train_border[7]))
            self.spoofed_index = 2500
            print("**********  Added class 3 and 6 for training *********")

        self.next()
        batch_data = self.current_batch
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch_data)))

        self.G = len(genomes)
        self.l_s_n = np.zeros((self.batch_size, self.G))
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
        generation_champions = []
        for champion_index in champion_indexes:
            genome_id, genome = genomes[champion_index]
            generation_champions.append(genome)
        jobs = []
        for genome in generation_champions:
            net = neat.nn.RecurrentNetwork.create(genome, config)
            jobs.append(self.pool.apply_async(evaluate, (net, self.validation_data)))

        validation_eer = np.zeros(10)

        index_grand_champion = 0
        for job, genome in zip(jobs, generation_champions):
            validation_eer[index_grand_champion] = job.get(timeout=self.timeout)
            index_grand_champion += 1
        if validation_eer.min() < self.best_eer_validation:
            self.gc = generation_champions[np.argmin(validation_eer)]
            self.best_eer_validation = validation_eer.min()
            self.app_gc = self.nb_generations

        self.validation_eer[self.nb_generations] = validation_eer

        self.nb_generations += 1

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
        if batch_size // 2 + self.spoofed_index >= len(self.spoofed_train) + 1:
            self.spoofed_index = 0
            rd.shuffle(self.spoofed_train)
        for index in range(self.batch_size // 2):
            self.current_batch.append(self.data[self.spoofed_train[self.spoofed_index + index]])

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
    l_s_n = np.zeros(batch_size)
    for data in batch_data:
        inputs, output = data[0], data[1]
        net.reset()
        xo = gate_lfcc(net, inputs)
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    for i in range(batch_size // 2):
        l_s_n[i] = (non_target_scores >= target_scores[i]).sum() / (batch_size // 2)

    for i in range(batch_size // 2):
        l_s_n[i + batch_size // 2] = (target_scores <= non_target_scores[i]).sum() / (batch_size // 2)

    return 1 - l_s_n


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
    # p.add_reporter(neat.Checkpointer(generation_interval=40, time_interval_seconds=None))

    # Run for up to n_gen generations.
    multi_evaluator = Anti_spoofing_Evaluator(n_processes, eval_genome, train_short_loader, validation_loader,
                                              pop_size=config_.pop_size)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner_))

    return winner_, config_, stats_, multi_evaluator.validation_eer, multi_evaluator.gc, multi_evaluator.app_gc, multi_evaluator.best_eer_validation


def evaluate(net, data_loader):
    """
    compute the eer equal error rate
    :param net: network
    :param data_loader: test dataset, contains audio files in a numpy array format
    :return eer
    """
    target_scores = []
    non_target_scores = []
    for data in tqdm(data_loader):
        net.reset()
        sample_input, output = data[0], data[1]
        xo = gate_lfcc(net, sample_input)
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

    train_short_loader = ASVDatasetshort(nb_samples=nb_samples_train, do_lfcc=True, do_standardize=True)
    validation_loader = ASVDatasetshort(do_lfcc=True, do_standardize=True, index_list=index_validation)
    train_loader = ASVDataset(is_train=True, is_eval=False, nb_samples=25380, do_lfcc=True, do_standardize=True)
    dev_loader = ASVDataset(is_train=False, is_eval=False, nb_samples=71237, do_lfcc=True, do_standardize=True)
    eval_loader = ASVDataset(is_train=False, is_eval=True, nb_samples=24844, do_lfcc=True, do_standardize=True)

    eer_list_train = []
    eer_list_dev = []
    eer_list_eval = []
    winner_list = []
    eer_list_train_gc = []
    eer_list_dev_gc = []
    eer_list_eval_gc = []
    gc_list = []
    stats_list = []
    validation_eer_array = np.zeros((n_iterations, n_generation, 10))
    app_gc_list = []
    best_eer_val_list = []

    for iterations in range(n_iterations):
        print("iterations number =", iterations)
        winner, config, stats, validation_eer, gc, app_gc, best_eer_val = run(config_path, n_generation)
        winner_net = neat.nn.RecurrentNetwork.create(winner, config)
        gc_net = neat.nn.RecurrentNetwork.create(gc, config)
        visualize.plot_stats(stats, ylog=False, view=True)

        eer_train = evaluate(winner_net, train_loader)
        eer_dev = evaluate(winner_net, dev_loader)
        eer_eval = evaluate(winner_net, eval_loader)

        eer_train_gc = evaluate(gc_net, train_loader)
        eer_dev_gc = evaluate(gc_net, dev_loader)
        eer_eval_gc = evaluate(gc_net, eval_loader)

        eer_list_train.append(eer_train)
        eer_list_dev.append(eer_dev)
        eer_list_eval.append(eer_eval)
        winner_list.append(winner)
        eer_list_train_gc.append(eer_train_gc)
        eer_list_dev_gc.append(eer_dev_gc)
        eer_list_eval_gc.append(eer_eval_gc)
        gc_list.append(gc)
        stats_list.append(stats)
        validation_eer_array[iterations, :, :] = validation_eer
        app_gc_list.append(app_gc)
        best_eer_val_list.append(best_eer_val)

    print("\n")
    print("equal error rate train", eer_list_train)
    show_stats(np.array(eer_list_train))

    print("\n")
    print("equal error rate dev", eer_list_dev)
    show_stats(np.array(eer_list_dev))

    print("\n")
    print("equal error rate eval", eer_list_eval)
    show_stats(np.array(eer_list_eval))

    print("\n")
    print("gc equal error rate train", eer_list_train_gc)
    show_stats(np.array(eer_list_train_gc))

    print("\n")
    print("gc equal error rate dev", eer_list_dev_gc)
    show_stats(np.array(eer_list_dev_gc))

    print("\n")
    print("gc equal error rate eval", eer_list_eval_gc)
    show_stats(np.array(eer_list_eval_gc))

    print("\n")
    print("gc equal error rate val", best_eer_val_list)
    show_stats(np.array(best_eer_val_list))

    print("\n")
    print("generations app gc", app_gc_list)
    show_stats_no_percentage(np.array(app_gc_list))
    
    validation_eer_array = 100 * validation_eer_array
    average_validation_eer = pd.DataFrame(validation_eer_array.mean(axis=2))
    min_validation_eer = pd.DataFrame(validation_eer_array.min(axis=2))

    sns.set(color_codes=True)
    sns.pointplot(data=average_validation_eer, estimator=np.median, color="red", label="average")
    sns.pointplot(data=min_validation_eer, estimator=np.median, label="min")
    plt.xlabel("generations")
    plt.ylabel("eer (%)")
    plt.title("Evolution of the validation eer over generations")
    plt.show()
