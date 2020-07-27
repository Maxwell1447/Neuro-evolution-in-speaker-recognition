import torch
from torch import sigmoid
import numpy as np
from neat_local.nn.recurrent_net import RecurrentNet
import neat

from anti_spoofing.metrics_utils import rocch2eer, rocch


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


def eval_genome_bce(g, conf, batch, return_correct=False):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    """

    # inputs: batch_size x t x bins
    # outputs: batch_size
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
        norm += confidence

    jitter = 1e-8
    prediction = (contribution / (norm + jitter))

    # return an array of True/False according to the correctness of the prediction
    if return_correct:
        return (prediction - outputs).abs() < 0.5

    # return the fitness computed from the BCE loss
    with torch.no_grad():
        return (1 / (1 + torch.nn.BCELoss()(prediction, outputs))).item()


def eval_genome_eer(g, conf, batch):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    Here the fitness function is the equal error rate
    """

    # inputs: batch_size x t x bins
    # outputs: batch_size
    inputs, outputs = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    target_scores = []
    non_target_scores = []

    jitter = 1e-8

    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset()
    for (input_t, output) in zip(inputs, outputs):
        # input_t: batch_size x bins

        # Usage of batch evaluation provided by PyTorch-NEAT
        xo = net.activate(input_t)  # batch_size x 2
        score = xo[:, 1]
        confidence = xo[:, 0]
        contribution = (score * confidence).sum() / (jitter + confidence).sum()

        if output == 1:
            target_scores.append(contribution)
        else:
            non_target_scores.append(contribution)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return 2 * (.5 - eer)
