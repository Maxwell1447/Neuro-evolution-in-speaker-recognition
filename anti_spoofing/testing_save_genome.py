import neat
import numpy as np
from tqdm import tqdm
import os
import pickle

from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.utils_ASV import whiten, gate_mfcc
from anti_spoofing.metrics_utils import rocch2eer, rocch


def evaluate(net, data_loader):
    """
    compute the eer equal error rate
    :param net: network
    :param data_loader: test dataset, contains audio files in a numpy array format
    :return eer
    """
    target_scores = []
    non_target_scores = []
    for data in tqdm(data_loader):
        net.reset()
        sample_input, output = data[0], data[1]
        sample_input = whiten(sample_input)
        xo = gate_mfcc(net, sample_input)
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return eer


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    train_set = ASVDataset(is_train=True, is_eval=False, nb_samples=25380, do_mfcc=True)
    dev_set = ASVDataset(is_train=False, is_eval=False, nb_samples=24844, do_mfcc=True)
    eval_set = ASVDataset(is_train=False, is_eval=True, nb_samples=80000, do_mfcc=True)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    winner = pickle.load(open('best_genome_eoc_batch_120_c3_balanced_test', 'rb'))

    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    train_eer = evaluate(winner_net, train_set)
    dev_eer = evaluate(winner_net, dev_set)
    eer = evaluate(winner_net, eval_set)

    print("\n")
    print("**** equal error rate train = {}  ****".format(train_eer))

    print("\n")
    print("**** equal error rate dev = {}  ****".format(dev_eer))

    print("\n")
    print("**** equal error rate = {}  ****".format(eer))

    """
    test_seen_classes = []
    test_unseen_classes = []

    for x in testset:
        if x[2] == 0:
            test_seen_classes.append(x)
            test_unseen_classes.append(x)
        elif x[2] in [16, 19]:
            test_seen_classes.append(x)
        else:
            test_unseen_classes.append(x)
    eer_seen = evaluate(winner_net, test_seen_classes)
    eer_unseen = evaluate(winner_net, test_unseen_classes)

    print("\n")
    print("**** equal error rate seen classes = {}  ****".format(eer_seen))

    print("\n")
    print("**** equal error rate unseen classes = {}  ****".format(eer_unseen))
    """

