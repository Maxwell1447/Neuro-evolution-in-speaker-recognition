import torch
from six import itervalues
from tqdm import tqdm
import numpy as np

from anti_spoofing.metrics_utils import rocch, rocch2eer
from neat_local.nn import RecurrentNet


def get_true_winner(config, population, data, max_batch=1):

    min_eer = 1.0
    best = None
    iter_data = iter(data)
    batches = [next(iter_data) for _ in range(max_batch)]
    for genome in tqdm(itervalues(population), total=config.pop_size):
        eer = evaluate_genome(genome, config, batches)
        if eer < min_eer:
            min_eer = eer
            best = genome

    return best


def get_best_candidates(population, n):
    fitnesses = np.array([g.fitness for g in itervalues(population)])
    print(fitnesses)
    n_best_idx = np.argsort(fitnesses)[-n:]
    n_best = []
    for i, g in enumerate(itervalues(population)):
        if i in n_best_idx:
            n_best.append(g)

    return n_best


def evaluate_genome(g, conf, batches):
    tgs = []
    ntgs = []
    for batch in batches:
        tg, ntg = evaluate_batch(g, conf, batch)
        tgs.append(tg)
        ntgs.append(ntg)
    target_scores = np.concatenate(tgs)
    non_target_scores = np.concatenate(ntgs)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer


def evaluate_batch(g, conf, batch):
    inputs, outputs = batch
    # inputs: t x batch_size x bins
    inputs = inputs.transpose(0, 1)

    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset()

    contribution = torch.zeros(len(outputs))
    norm = torch.zeros(len(outputs))
    for input_t in inputs:
        # input_t: batch_size x bins

        xo = net.activate(input_t)  # batch_size x 2
        score = xo[:, 1]
        confidence = xo[:, 0]
        contribution += score * confidence  # batch_size
        norm += confidence  # batch_size

    jitter = 1e-8
    prediction = contribution / (norm + jitter)  # batch_size

    target_scores = prediction[outputs == 1].numpy()  # select with mask when target == 1
    non_target_scores = prediction[outputs == 0].numpy()  # select with mask when target == 0

    return target_scores, non_target_scores
