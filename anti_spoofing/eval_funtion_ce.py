import torch
from torch import sigmoid
import numpy as np
from neat_local.nn.recurrent_net import RecurrentNet
import neat


class ProcessedASVEvaluator(neat.parallel.ParallelEvaluator):
    """
    Allows parallel batch evaluation using an Iterator model with next().
    The eval function itself is not defined here.
    """

    def __init__(self, num_workers, eval_function, data, timeout=None):
        super().__init__(num_workers, eval_function, timeout)
        self.data = data  # PyTorch DataLoader
        self.data_iter = iter(data)
        self.timeout = timeout

    def evaluate(self, genomes, config):
        batch = self.next()
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)

    def next(self):
        try:
            batch = next(self.data_iter)
            return batch
        except StopIteration:
            self.data_iter = iter(self.data)
        return next(self.data_iter)


def eval_genome_ce(g, conf, batch):
    """
    This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    Here the fitness function is the cross entropy with softmax
    """

    # inputs: batch_size x t x bins
    # outputs: batch_size
    inputs, _, outputs = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    jitter = 1e-8
    cross_entropy = 0
    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset()
    for (input_t, output) in zip(inputs, outputs):
        # input_t: batch_size x bins

        # Usage of batch evaluation provided by PyTorch-NEAT
        xo = net.activate(input_t)  # batch_size x 8
        score = xo[:, 1:].reshape(torch.Size((7, -1)))
        confidence = sigmoid(xo[:, 0])
        contribution = (score * confidence).sum(axis=1) / (jitter + confidence).sum()
        contribution[np.isinf(contribution)] = 100
        contribution[np.isnan(contribution)] = 100
        scores = contribution.exp()
        scores = scores / scores.sum()
        cross_entropy -= np.log(scores[int(output.item())] + jitter)

    batch_size = list(outputs.size())[0]
    normalize_fitness = - batch_size * np.log(1 / 7)

    return float(1 - cross_entropy / normalize_fitness)
