import torch
from torch.utils.data.dataloader import DataLoader
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

    def __init__(self, num_workers, eval_function, data, timeout=None, backprop=False,
                 batch_increment=0, initial_batch_size=100,
                 batch_generations=50, use_gate=True):
        if num_workers > 1:
            super().__init__(num_workers, eval_function, timeout)
        else:
            self.eval_function = eval_function
            self.num_workers = 1
        self.use_gate = use_gate
        self.data = data  # PyTorch DataLoader / Dataset
        self.batch_size = initial_batch_size
        self.batch_increment = batch_increment
        self.gen = 0
        self.batch_generation = batch_generations
        if self.batch_increment > 0:
            data = DataLoader(self.data, batch_size=self.batch_size, num_workers=4, shuffle=True, drop_last=True)

        self.data_iter = iter(data)
        self.timeout = timeout
        self.backprop = backprop

    def __del__(self):
        if self.num_workers > 1:
            self.pool.close()  # should this be terminate?
            self.pool.join()

    def evaluate(self, genomes, config):
        batch = self.next()
        jobs = []
        if self.num_workers == 1:
            for _, genome in genomes:
                genome.fitness = self.eval_function(genome, config, batch, self.backprop, self.use_gate)
        else:
            for _, genome in genomes:
                jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch, self.backprop,
                                                                       self.use_gate)))

            # assign the fitness back to each genome
            for job, (_, genome) in zip(jobs, genomes):
                genome.fitness = job.get(timeout=self.timeout)

    def next(self):
        try:
            batch = next(self.data_iter)
            if self.batch_increment > 0 and self.gen >= self.batch_generation:
                raise StopIteration
            self.gen += 1
            return batch
        except StopIteration:
            if self.batch_increment > 0:
                self.batch_size += self.batch_increment
                print("*****************")
                print(self.batch_size)
                print("*****************")
                self.data_iter = iter(DataLoader(self.data, batch_size=self.batch_size,
                                                 num_workers=4, shuffle=True, drop_last=True))
                self.gen = 0
            else:
                self.data_iter = iter(self.data)
        self.gen += 1
        return next(self.data_iter)


def eval_efficiency(g, conf):

    connections, useful_connections = backprop_neat.required_connections(g, conf)

    return len(useful_connections) / len(connections)


def eval_genome_bce(g, conf, batch, backprop, use_gate, return_correct=False, efficiency_contribution=0.):
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

    if backprop and not conf.genome_config.backprop:
        ctx = torch.no_grad()
        ctx.__enter__()
    else:
        ctx = None
    net.reset(len(targets))
    contribution = torch.zeros(len(targets))
    norm = torch.zeros(len(targets))
    for input_t in inputs:
        # input_t: batch_size x bins

        # Usage of batch evaluation provided by PyTorch-NEAT
        xo = net.activate(input_t)  # batch_size x 2
        score = xo[:, 1]
        confidence = xo[:, 0] if use_gate else torch.ones_like(score)
        contribution += score * confidence  # batch_size
        norm += confidence

    jitter = 1e-8
    prediction = (contribution / (norm + jitter))

    # return an array of True/False according to the correctness of the prediction
    if return_correct:
        return (prediction - targets).abs() < 0.5

    if backprop and conf.genome_config.backprop:
        optimizer = torch.optim.SGD(g.get_params(), lr=conf.genome_config.learning_rate)
        loss = torch.nn.BCELoss()(prediction, targets)
        loss.backward()
        optimizer.step()
        g.clamp(conf.genome_config)

        if efficiency_contribution:
            assert 0 <= efficiency_contribution <= 1.
            fitness = 1 / (1 + loss.detach().item())
            efficiency = eval_efficiency(g, conf)

            return efficiency_contribution * efficiency + (1 - efficiency_contribution) * fitness
        else:
            return 1 / (1 + loss.detach().item())

    elif backprop:

        loss = torch.nn.BCELoss()(prediction, targets)
        fitness = 1 / (1 + loss.detach().item())
        if backprop and not conf.genome_config.backprop:
            ctx.__exit__()
        if efficiency_contribution:
            assert 0 <= efficiency_contribution <= 1.
            fitness = 1 / (1 + loss.detach().item())
            efficiency = eval_efficiency(g, conf)

            return efficiency_contribution * efficiency + (1 - efficiency_contribution) * fitness
        else:
            return fitness

    # return the fitness computed from the BCE loss
    with torch.no_grad():
        return 1 / (1 + torch.nn.BCELoss()(prediction, targets).item())


def eval_genome_eer(g, conf, batch, backprop=False, use_gate=True):
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
        confidence = xo[:, 0] if use_gate else torch.ones_like(score)
        contribution += score * confidence  # batch_size
        norm += confidence  # batch_size

    jitter = 1e-8
    prediction = contribution / (norm + jitter)  # batch_size

    target_scores = prediction[targets == 1].numpy()  # select with mask when target == 1
    non_target_scores = prediction[targets == 0].numpy()  # select with mask when target == 0

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return 2 * (.5 - eer)


def feed_and_predict(data, g, conf, backprop, use_gate=True, loading_bar=True):
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

        if loading_bar:
            iterable = tqdm(data_iter, total=len(data))
        else:
            iterable = data_iter

        for batch in iterable:

            input, target = batch  # input: batch x t x BIN; output: batch
            net.reset(len(target))
            input = input.transpose(0, 1)  # input: t x batch x BIN
            batch_size = target.shape[0]
            norm = torch.zeros(batch_size)
            contribution = torch.zeros(batch_size)
            for input_t in input:
                xo = net.activate(input_t)  # batch x 2
                score = xo[:, 1]  # batch
                confidence = xo[:, 0] if use_gate else torch.ones_like(score)
                contribution += score * confidence  # batch
                norm += confidence  # batch
            prediction = contribution / (norm + jitter)  # batch

            predictions.append(prediction.numpy())
            targets.append(target.numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    return predictions, targets


def evaluate_eer_acc(g, conf, data, backprop=False, use_gate=True, loading_bar=True):
    """
    returns the equal error rate and the accuracy
    """

    predictions, targets = feed_and_predict(data, g, conf, backprop, use_gate=use_gate, loading_bar=loading_bar)

    accuracy = np.mean(np.abs(predictions - targets) < 0.5)

    target_scores = predictions[targets == 1]
    non_target_scores = predictions[targets == 0]

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer, accuracy
