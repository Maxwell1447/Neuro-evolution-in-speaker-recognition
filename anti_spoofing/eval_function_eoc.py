import torch
from torch import sigmoid
import numpy as np
from neat_local.nn.recurrent_net import RecurrentNet
import neat


class ProcessedASVEvaluatorEoc2(neat.parallel.ParallelEvaluator):
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

        _, outputs = batch

        self.G = len(genomes)
        self.l_s_n = np.empty((len(outputs), self.G))

        pseudo_genome_id = 0
        # return ease of classification for each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            self.l_s_n[:, pseudo_genome_id] = job.get(timeout=self.timeout)
            pseudo_genome_id += 1

        # compute the fitness
        p_s = np.sum(self.l_s_n, axis=1).reshape(-1, 1) / self.G

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
        F = np.sum(self.l_s_n.reshape(p_s.size, -1) * (1 - p_s), axis=0) / (np.sum(1 - p_s) + 1e-8)

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
    if len(batch) == 3:
        inputs, outputs, _ = batch
    else:
        inputs, outputs = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset()

    contribution = torch.zeros(len(outputs))
    norm = torch.zeros(len(outputs))
    for input_t in inputs:
        # input_t: batch_size x bins

        # Usage of batch evaluation provided by PyTorch-NEAT
        xo = net.activate(input_t)  # batch_size x 2
        score = xo[:, 1]
        confidence = xo[:, 0]
        contribution += score * confidence  # batch_size
        norm += confidence  # batch_size

    jitter = 1e-8
    prediction = contribution / (norm + jitter)          # batch_size

    target_scores = prediction[outputs == 1].numpy()  # bonafide scores
    non_target_scores = prediction[outputs == 0].numpy()  # spoofed scores

    l_s_n = np.empty_like(prediction)
    prediction = prediction.numpy()

    for i, (pred, out) in enumerate(zip(prediction, outputs)):

        if out:  # if bonafide
            l_s_n[i] = 1 - np.sum(non_target_scores >= pred) / non_target_scores.size
        else:  # if spoofed
            l_s_n[i] = 1 - np.sum(target_scores <= pred) / target_scores.size

    return l_s_n
