import os
import shutil
import sys

from six import itervalues
from torch.utils.tensorboard import SummaryWriter

import torch

from anti_spoofing.data_loader import load_data
from anti_spoofing.eval_functions import *
from anti_spoofing.constants import *

import backprop_neat as neat
from backprop_neat import required_connections

USE_DATASET = False
if sys.platform.find("win") >= 0:
    DATA_ROOT = './data'
else:
    DATA_ROOT = os.path.join("..", "..", "..", "speechmaterials", "databases", "ASVspoof")

CHECKPOINT = 9999


def custom_genome(config, n_hidden=20):

    genome = config.genome_type(0)
    genome.configure_new(config.genome_config)

    for key_node in range(2, n_hidden + 2):
        genome.nodes[key_node] = genome.create_node(config.genome_config, key_node)

    nodes = config.genome_config.input_keys + list(genome.nodes.keys())
    keys_in = config.genome_config.input_keys
    keys_hidden = list(range(2, n_hidden + 2))
    keys_out = [0, 1]

    # for key_in in nodes:
    #     for key_out in genome.nodes:
    #         key = (key_in, key_out)
    #         genome.connections[key] = genome.create_connection(config.genome_config, key_in, key_out)

    for key_in in keys_in:
        for key_hidden in keys_hidden:
            k = (key_in, key_hidden)
            genome.connections[k] = genome.create_connection(config.genome_config, key_in, key_hidden)

    for key_out in keys_out:
        for key_hidden in keys_hidden:
            k = (key_hidden, key_out)
            genome.connections[k] = genome.create_connection(config.genome_config, key_hidden, key_out)

    return genome


def get_eer_acc(y, out):
    """
    returns the equal error rate and the accuracy
    """

    target_scores = out[y == 1].numpy()
    non_target_scores = out[y == 0].numpy()

    acc = ((y - out).abs() < 0.5).float().mean()

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer, acc


def loss_bce(genome, batch, use_gate, net=None):
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

    if net is None:
        net = backprop_neat.nn.RecurrentNet.create(genome, config, device="cpu", dtype=torch.float32)
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

    return prediction, targets


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'ASV_neat_preprocessed_backprop.cfg')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    train_data, devloader, evalloader = load_data(batch_size=100, length=3 * 16000, num_train=10000,
                                                  custom_path=DATA_ROOT, multi_proc=False, balanced=True,
                                                  batch_size_test=100, include_eval=True,
                                                  return_dataset=USE_DATASET)

    p, _ = neat.Checkpointer.restore_checkpoint("neat-checkpoint_-{}".format(CHECKPOINT))

    max_fitness = 0
    best_genome = None
    for g in itervalues(p.population):
        if g.fitness is not None and g.fitness > max_fitness:
            best_genome = g
            max_fitness = g.fitness

    genome = best_genome

    # genome = custom_genome(config)

    # co, useful_co = required_connections(genome, config)
    #
    # print(len(useful_co))
    # print(len(useful_co)/len(co))

    key = list(genome.connections)[3]
    keys = list(genome.connections)[:5]

    optimizer = torch.optim.Adam(genome.get_params(), lr=0.01)
    criterion = torch.nn.BCELoss()
    if not os.path.isdir('./runs/NEAT_pure_backprop'):
        os.makedirs('./runs/NEAT_pure_backprop')
    shutil.rmtree('./runs/NEAT_pure_backprop')
    writer = SummaryWriter("./runs/NEAT_pure_backprop")
    i = 0
    # net = backprop_neat.nn.FeedForwardNetwork.create(genome, config)

    for epoch in range(10):
        print("EPOCH {}".format(epoch))
        for batch in tqdm(iter(train_data)):
            optimizer.zero_grad()

            prediction, targets = loss_bce(genome, batch, True)

            loss = criterion(prediction, targets)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                eer, acc = get_eer_acc(targets, prediction)

            writer.add_scalar("loss", loss.detach().item(), i)
            writer.add_scalar("train eer", eer, i)
            writer.add_scalar("train acc", acc, i)

            for k in keys:
                writer.add_scalar("weight_{}".format(k), genome.connections[k].weight, i)
            i += 1

    with torch.no_grad():
        dev_eer, dev_acc = evaluate_eer_acc(genome, config, devloader)
        eval_eer, eval_acc = evaluate_eer_acc(genome, config, evalloader)

    print("dev EER / ACC")
    print(dev_eer)
    print(dev_acc)
    print()
    print("-------------")

    print("eval EER / ACC")
    print(eval_eer)
    print(eval_acc)
