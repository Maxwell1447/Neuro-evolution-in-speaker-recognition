import torch
from six import itervalues
from torch import sigmoid
import numpy as np
from tqdm import tqdm

import backprop_neat
import neat_local
import neat_local.nn

from torch.utils.data import DataLoader

from anti_spoofing.eval_functions import ProcessedASVEvaluator

import neat
from anti_spoofing.metrics_utils import rocch2eer, rocch
from time import time
from timeit import timeit
from copy import copy


class ProcessedASVEvaluatorEoc(ProcessedASVEvaluator):
    """
    Allows parallel batch evaluation using an Iterator model with next().
    The eval function itself is not defined here.
    """

    def __init__(self, num_workers, eval_function, data, pop, timeout=None, batch_increment=0, initial_batch_size=100,
                 batch_generations=50, backprop=False, use_gate=True):
        super().__init__(num_workers, eval_function, data, timeout=timeout, batch_increment=batch_increment,
                         initial_batch_size=initial_batch_size, batch_generations=batch_generations, backprop=backprop,
                         use_gate=use_gate)

        self.G = pop
        self.backprop = backprop
        self.l_s_n = []

    def evaluate(self, genomes, config):
        batch = self.next()
        jobs = []



        _, outputs = batch

        self.G = len(genomes)
        self.l_s_n = torch.empty((len(outputs), self.G))

        # return ease of classification for each genome
        if self.backprop:
            for i, (_, genome) in enumerate(genomes):
                self.l_s_n[:, i] = self.eval_function(genome, config, batch, self.backprop, self.use_gate)
        else:
            for ignored_genome_id, genome in genomes:
                jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch, self.backprop,
                                                                       self.use_gate)))
            results = [job.get(timeout=self.timeout) for job in jobs]
            for i, genome in enumerate(genomes):
                self.l_s_n[:, i] = results[i]

        # compute the fitness
        p_s = torch.sum(self.l_s_n, dim=1).view(-1, 1) / self.G

        F = torch.sum(self.l_s_n * (torch.tensor(1.) - p_s), dim=0) \
            / (torch.sum(torch.tensor(1.) - p_s) + 1e-8)

        pseudo_genome_id = 0
        # assign the fitness back to each genome
        for ignored_genome_id, genome in tqdm(genomes):
            if self.backprop:
                optimizer = torch.optim.SGD(genome.get_params(), lr=config.genome_config.learning_rate)
                optimizer.zero_grad()
                loss = 1 - F[pseudo_genome_id]
                loss.backward(retain_graph=True)
                optimizer.step()

                genome.fitness = F[pseudo_genome_id].detach().item()
            else:
                genome.fitness = F[pseudo_genome_id].item()
            pseudo_genome_id += 1


class ProcessedASVEvaluatorEocGc(ProcessedASVEvaluatorEoc):
    """
    Allows parallel batch evaluation using an Iterator model with next().
    The eval function itself is not defined here.
    """

    def __init__(self, num_workers, eval_function, data, validation_data, pop, gc_eval, config, timeout=None,
                 batch_increment=0, initial_batch_size=100,
                 batch_generations=50, backprop=False, use_gate=True):
        super().__init__(num_workers, eval_function, data, timeout=timeout, batch_increment=batch_increment,
                         initial_batch_size=initial_batch_size, batch_generations=batch_generations, pop=pop,
                         backprop=backprop, use_gate=use_gate)

        self.validation_data = validation_data
        self.val_data_iter = iter(validation_data)
        self.gc_eval = gc_eval
        self.config = config
        self.backprop = backprop
        self.gc = None
        self.eer_gc = 1
        self.generations = 0
        self.app_gc = 0

    def evaluate(self, genomes, config):
        batch = self.next()
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, batch,
                                                                   self.backprop, self.use_gate)))

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

        batch = self.next_val()
        champions_eer = np.zeros(10)

        jobs = []
        for genome in generation_champions:
            jobs.append(self.pool.apply_async(self.gc_eval, (genome, config, batch)))

        index_grand_champion = 0
        for job, genome in zip(jobs, generation_champions):
            champions_eer[index_grand_champion] = job.get(timeout=self.timeout)
            index_grand_champion += 1
        if champions_eer.min() < self.eer_gc:
            self.gc = generation_champions[np.argmin(champions_eer)]
            self.eer_gc = champions_eer.min()
            self.app_gc = self.generations
        self.generations += 1

    def next_val(self):
        try:
            batch = next(self.val_data_iter)
            return batch
        except StopIteration:
            self.val_data_iter = iter(self.validation_data)
        return next(self.val_data_iter)


def eval_genome_eoc(g, conf, batch, backprop, use_gate):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    Here the fitness function is the ease of classification
    """
    assert not backprop
    # inputs: batch_size x t x bins
    # outputs: batch_size
    if len(batch) == 3:
        inputs, targets, _ = batch
    else:
        inputs, targets = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    net = neat_local.nn.RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
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
        norm += confidence  # batch_size

    jitter = 1e-8
    prediction = contribution / (norm + jitter)  # batch_size

    target_scores = prediction[targets == 1]  # bonafide scores
    non_target_scores = prediction[targets == 0]  # spoofed scores

    l_s_n = torch.empty_like(prediction)

    for i, (pred, out) in enumerate(zip(prediction, targets)):

        if out:  # if bonafide
            l_s_n[i] = torch.tensor(1.) - torch.sum((non_target_scores >= pred).float()) / non_target_scores.shape[0]
        else:  # if spoofed
            l_s_n[i] = torch.tensor(1.) - torch.sum((target_scores <= pred).float()) / target_scores.shape[0]

    return l_s_n


def quantified_eval_genome_eoc(g, conf, batch, backprop, use_gate):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    Here the fitness function is the quantified ease of classification
    """
    assert not backprop
    # inputs: batch_size x t x bins
    # outputs: batch_size
    if len(batch) == 3:
        inputs, targets, _ = batch
    else:
        inputs, targets = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    net = neat_local.nn.RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
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
        norm += confidence  # batch_size

    jitter = 1e-8
    prediction = contribution / (norm + jitter)  # batch_size

    target_scores = prediction[targets == 1]  # bonafide scores
    non_target_scores = prediction[targets == 0] # spoofed scores

    l_s_n = torch.empty_like(prediction)

    for i, (pred, out) in enumerate(zip(prediction, targets)):

        if out:  # if bonafide
            l_s_n[i] = torch.tensor(1.) - torch.sum(non_target_scores[non_target_scores >= pred] - pred + 1) \
                       / torch.sum(torch.abs(non_target_scores - pred) + 1)
        else:  # if spoofed
            l_s_n[i] = torch.tensor(1.) - torch.sum(pred - target_scores[target_scores <= pred] + 1) \
                       / torch.sum(torch.abs(target_scores - pred) + 1)

    return l_s_n


def double_quantified_eval_genome_eoc(g, conf, batch, backprop, use_gate):
    """
    Same than eval_genomes() but for 1 genome. This function is used for parallel evaluation.
    The input is already preprocessed with shape batch_size x t x bins
    t: index of the windows used for the pre-processing
    bins: number of features extracted --> corresponds to the number of input neurons of the recurrent net
    Here the fitness function is the double quantified ease of classification
    """

    # inputs: batch_size x t x bins
    # outputs: batch_size
    if len(batch) == 3:
        inputs, targets, _ = batch
    else:
        inputs, targets = batch
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
        confidence = xo[:, 0] if use_gate else torch.ones_like(score)
        contribution += score * confidence  # batch_size
        norm += confidence  # batch_size

    jitter = 1e-8
    prediction = contribution / (norm + jitter)  # batch_size

    target_scores = prediction[targets == 1]  # bonafide scores
    non_target_scores = prediction[targets == 0]  # spoofed scores

    l_s_n = torch.empty_like(prediction)

    for i, (pred, out) in enumerate(zip(prediction, targets)):

        if out:  # if bonafide
            l_s_n[i] = (torch.sum(pred - non_target_scores[non_target_scores <= pred] + 1)
                        - torch.sum(non_target_scores[non_target_scores >= pred] - pred + 1)) \
                       / torch.sum(torch.abs(non_target_scores - pred) + 1)
        else:  # if spoofed
            l_s_n[i] = (torch.sum(target_scores[target_scores >= pred] - pred + 1)
                        - torch.sum(pred - target_scores[target_scores <= pred] + 1)) \
                       / torch.sum(torch.abs(target_scores - pred) + 1)

    return l_s_n


def eval_eer_gc(g, config, batch):
    jitter = 1e-8
    net = neat_local.nn.RecurrentNet.create(g, config, device="cpu", dtype=torch.float32)
    net.reset()
    input, output = batch  # input: batch x t x BIN; output: batch
    input = input.transpose(0, 1)  # input: t x batch x BIN
    batch_size = output.shape[0]
    norm = torch.zeros(batch_size)
    contribution = torch.zeros(batch_size)
    for input_t in input:
        xo = net.activate(input_t)  # batch x 2
        score = xo[:, 1]  # batch
        confidence = xo[:, 0]  # batch
        contribution += score * confidence  # batch
        norm += confidence  # batch

    predictions = contribution / (norm + jitter)  # batch

    target_scores = predictions[output == 1]
    non_target_scores = predictions[output == 0]

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)
    return eer
