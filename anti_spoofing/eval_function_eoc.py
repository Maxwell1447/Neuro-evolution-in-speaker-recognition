import torch
from torch import sigmoid
import numpy as np
from neat_local.nn.recurrent_net import RecurrentNet
import neat

from anti_spoofing.metrics_utils import rocch2eer, rocch


class ProcessedASVEvaluatorEoc(neat.parallel.ParallelEvaluator):
    """
    Allows parallel batch evaluation using an Iterator model with next().
    The eval function itself is not defined here.
    """

    def __init__(self, num_workers, eval_function, data, pop, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.data = data  # PyTorch DataLoader
        self.data_iter = iter(data)
        self.timeout = timeout
        self.G = pop
        self.l_s_n = []

    def evaluate(self, genomes, config):
        batch = self.next()
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch)))

        self.G = len(genomes)
        self.l_s_n = []

        pseudo_genome_id = 0
        # return ease of classification for each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            self.l_s_n.append(job.get(timeout=self.timeout))
            pseudo_genome_id += 1

        # compute the fitness
        self.l_s_n = np.array(self.l_s_n)

        p_s = np.sum(self.l_s_n, axis=0).reshape(-1, 1) / self.G

        F = np.sum(self.l_s_n.reshape(p_s.size, -1) * (1 - p_s), axis=0) / np.sum(1 - p_s)

        pseudo_genome_id = 0
        # assign the fitness back to each genome
        for ignored_genome_id, genome in genomes:
            genome.fitness = F[pseudo_genome_id]
            pseudo_genome_id += 1

    def next(self):
        try:
            batch = next(self.data_iter)
            return batch
        except StopIteration:
            self.data_iter = iter(self.data)
        return next(self.data_iter)


class ProcessedASVEvaluatorEocGc(neat.parallel.ParallelEvaluator):
    """
    Allows parallel batch evaluation using an Iterator model with next().
    The eval function itself is not defined here.
    """

    def __init__(self, num_workers, eval_function, data, validation_data, pop, gc_eval, config, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.data = data  # PyTorch DataLoader
        self.data_iter = iter(data)
        self.timeout = timeout
        self.G = pop
        self.l_s_n = []
        self.validation_data = validation_data
        self.gc_eval = gc_eval
        self.config = config

    def evaluate(self, genomes, config):
        batch = self.next()
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch)))

        self.G = len(genomes)
        self.l_s_n = []

        pseudo_genome_id = 0
        # return ease of classification for each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            self.l_s_n.append(job.get(timeout=self.timeout))
            pseudo_genome_id += 1

        # compute the fitness
        self.l_s_n = np.array(self.l_s_n)
        p_s = np.sum(self.l_s_n, axis=0).reshape(-1, 1) / self.G
        F = np.sum(self.l_s_n.reshape(p_s.size, -1) * (1 - p_s), axis=0) / np.sum(1 - p_s)

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
            champions_eer[index_grand_champion] = self.gc_eval(genome, self.config, self.validation_data)
            index_grand_champion += 1

        grand_champion = genomes[champion_indexes[np.argmax(champions_eer)]]
        self.gc.append(grand_champion)

    def next(self):
        try:
            batch = next(self.data_iter)
            return batch
        except StopIteration:
            self.data_iter = iter(self.data)
        return next(self.data_iter)


def eval_genome_eoc(g, conf, batch):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    Here the fitness function is the ease of classification
    """

    # inputs: batch_size x t x bins
    # outputs: batch_size
    inputs, outputs = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    batch_size = len(batch)

    target_scores = []
    non_target_scores = []

    jitter = 1e-8

    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset()
    for (input_t, output) in zip(inputs, outputs):
        # input_t: batch_size x bins

        # Usage of batch evaluation provided by PyTorch-NEAT
        xo = sigmoid(net.activate(input_t))  # batch_size x 2
        score = xo[:, 1]
        confidence = xo[:, 0]
        contribution = (score * confidence).sum() / (jitter + confidence).sum()

        if output == 1:
            target_scores.append(contribution)
        else:
            non_target_scores.append(contribution)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    l_s_n = np.zeros(target_scores.size)

    for i in range(target_scores.size):
        l_s_n[i] = (non_target_scores >= target_scores[i]).sum() / non_target_scores.size

    return 1 - l_s_n
