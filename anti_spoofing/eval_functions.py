import torch
from torch import sigmoid
import numpy as np
from tqdm import tqdm

import neat_local.nn
import backprop_neat
import neat

from anti_spoofing.metrics_utils import rocch2eer, rocch


class ProcessedASVEvaluator(neat.parallel.ParallelEvaluator):
    """
    Allows parallel batch evaluation using an Iterator model with next().
    The eval function itself is not defined here.
    """

    def __init__(self, num_workers, eval_function, data, timeout=None, backprop=False):
        super().__init__(num_workers, eval_function, timeout)
        self.data = data  # PyTorch DataLoader
        self.data_iter = iter(data)
        self.timeout = timeout
        self.backprop = backprop

    def evaluate(self, genomes, config):
        batch = self.next()
        jobs = []

        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch, self.backprop)))

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


def eval_genome_bce(g, conf, batch, backprop, return_correct=False):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    """

    # inputs: batch_size x t x bins
    # outputs: batch_size
    inputs, targets = batch
    targets = targets.float()
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    if backprop:
        net = backprop_neat.nn.RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    else:
        net = neat_local.nn.RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset(len(targets))
    contribution = torch.zeros(len(targets))
    norm = torch.zeros(len(targets))
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
        return (prediction - targets).abs() < 0.5

    if backprop:
        optimizer = torch.optim.SGD(g.get_params(), lr=conf.genome_config.learning_rate)
        loss = torch.nn.BCELoss()(prediction, targets)
        loss.backward()
        optimizer.step()

        return 1 / (1 + loss.detach().item())

    # return the fitness computed from the BCE loss
    with torch.no_grad():
        return 1 / (1 + torch.nn.BCELoss()(prediction, targets).item())


def eval_genome_eer(g, conf, batch, backprop):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    Here the fitness function is the equal error rate
    """

    # inputs: batch_size x t x bins
    # outputs: batch_size
    inputs, targets = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    net = neat_local.nn.RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    assert not backprop
    net.reset(len(targets))

    contribution = torch.zeros(len(targets))
    norm = torch.zeros(len(targets))
    for input_t in inputs:
        # input_t: batch_size x bins

        xo = net.activate(input_t)  # batch_size x 2
        score = xo[:, 1]
        confidence = xo[:, 0]
        contribution += score * confidence  # batch_size
        norm += confidence  # batch_size

    jitter = 1e-8
    prediction = contribution / (norm + jitter)  # batch_size

    target_scores = prediction[targets == 1].numpy()  # select with mask when target == 1
    non_target_scores = prediction[targets == 0].numpy()  # select with mask when target == 0

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return 2 * (.5 - eer)


def feed_and_predict(data, g, conf, backprop):
    """
    returns predictions + targets in 2 numpy arrays
    """
    data_iter = iter(data)

    jitter = 1e-8

    if backprop:
        net = backprop_neat.nn.RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    else:
        net = neat_local.nn.RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(data_iter, total=len(data)):

            input, target = batch  # input: batch x t x BIN; output: batch
            net.reset(len(target))
            input = input.transpose(0, 1)  # input: t x batch x BIN
            batch_size = target.shape[0]
            norm = torch.zeros(batch_size)
            contribution = torch.zeros(batch_size)
            for input_t in input:
                xo = net.activate(input_t)  # batch x 2
                score = xo[:, 1]  # batch
                confidence = xo[:, 0]  # batch
                contribution += score * confidence  # batch
                norm += confidence  # batch
            prediction = contribution / (norm + jitter)  # batch

            predictions.append(prediction.numpy())
            targets.append(target.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    return predictions, targets


def evaluate_eer_acc(g, conf, data, backprop=False):
    """
    returns the equal error rate and the accuracy
    """

    predictions, targets = feed_and_predict(data, g, conf, backprop)

    accuracy = np.mean(np.abs(predictions - targets) < 0.5)

    target_scores = predictions[targets == 1]
    non_target_scores = predictions[targets == 0]

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer, accuracy
