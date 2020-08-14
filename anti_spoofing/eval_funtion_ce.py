import torch
from torch import sigmoid
import numpy as np
from neat_local.nn.recurrent_net import RecurrentNet
import neat
from tqdm import tqdm


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
    batch_size = outputs.shape[0]
    cross_entropy = 0
    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)
    net.reset()

    contribution = torch.zeros((len(outputs)), 7)
    norm = torch.zeros(len(outputs))

    for input_t in inputs:
        # input_t: batch_size x bins

        # Usage of batch evaluation provided by PyTorch-NEAT
        xo = net.activate(input_t)  # batch_size x 8
        score = xo[:, 1:]
        confidence = sigmoid(xo[:, 0])
        contribution += score * confidence.reshape((-1, 1))  # batch_size x 7
        norm += confidence  # batch_size x 7

    prediction = contribution / (norm.reshape((batch_size, 1)) + jitter)  # batch_size x 7

    prediction = prediction.exp()
    prediction = prediction / prediction.sum(axis=1).reshape((-1, 1))

    for index_audio in range(batch_size):
        cross_entropy -= np.log(prediction[index_audio][int(outputs[index_audio].item())] + jitter)

    return float(1 / (1 + cross_entropy))


def feed_and_predict_ce(data, g, conf):
    """
    returns predictions + targets in 2 numpy arrays
    """
    data_iter = iter(data)

    jitter = 1e-8

    net = RecurrentNet.create(g, conf, device="cpu", dtype=torch.float32)

    predictions = []
    targets = []

    for batch in tqdm(data_iter, total=len(data)):
        net.reset()
        input, _, output = batch  # input: batch x t x BIN; output: batch
        input = input.transpose(0, 1)  # input: t x batch x BIN
        batch_size = output.shape[0]
        contribution = torch.zeros((len(output)), 7)
        norm = torch.zeros(len(output))
        for input_t in input:
            xo = net.activate(input_t)  # batch x 8
            score = xo[:, 1:]  # batch x 7
            confidence = sigmoid(xo[:, 0])  # batch
            contribution += score * confidence.reshape((-1, 1))  # batch x 7
            norm += confidence  # batch
        prediction = contribution / (norm.reshape((batch_size, 1)) + jitter)  # batch x 7

        predictions.append(prediction.numpy())
        targets.append(output.numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    return predictions, targets
