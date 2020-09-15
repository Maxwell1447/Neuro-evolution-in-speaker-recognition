import neat
import numpy as np
from tqdm import tqdm
import os
import pickle

from anti_spoofing.data_utils import ASVDataset, ASVFile
from anti_spoofing.utils_ASV import whiten, compute_eer, gate_mfcc, gate_lfcc

separate_known_unknown = False
is_whiten = False
train_class_attack = list(range(1, 7))
eval_class_attack = list(range(7, 20))


def evaluate(net, data_loader, is_whiten=False):
    """
    compute the eer (equal error rate)
    :param is_whiten: If True, it will standardize the data_loader
    :param net: network
    :param data_loader: dataset, contains audio files or extracted features in a numpy array format
    :return eer
    """
    target_scores = []
    non_target_scores = []
    for data in tqdm(data_loader):
        net.reset()
        sample_input, output = data[0], data[1]
        if is_whiten:
            sample_input = whiten(sample_input)
        xo = gate_lfcc(net, sample_input)
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)

    eer = compute_eer(target_scores, non_target_scores)

    return eer


def evaluate_per_attack(net, data_loader, class_attack_list, is_whiten=False):
    """
    compute the eer (equal error rate)
    :param is_whiten: If True, it will standardize the data_loader
    :param net: network
    :param class_attack_list: list of the spoofed class
    :param data_loader: dataset, contains audio files or extracted features in a numpy array format
    :return eer
    """
    eer_list = []
    for class_attack in class_attack_list:
        data_one_class_attack = []
        for x in data_loader:
            if x[2] == class_attack or x[2] == 0:
                data_one_class_attack.append(x)
        eer_list.append(evaluate(net, data_one_class_attack, is_whiten=is_whiten))
    return eer_list


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    train_set = ASVDataset(is_train=True, is_eval=False, nb_samples=25380, do_standardize=True, do_lfcc=True)
    dev_set = ASVDataset(is_train=False, is_eval=False, nb_samples=24844, do_standardize=True, do_lfcc=True)
    eval_set = ASVDataset(is_train=False, is_eval=True, nb_samples=71238, do_standardize=True,  do_lfcc=True)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    winner = pickle.load(open('best_genome_eoc_class_4_lfcc', 'rb'))

    winner_net = neat.nn.RecurrentNetwork.create(winner, config)

    train_eer = evaluate(winner_net, train_set)
    dev_eer = evaluate(winner_net, dev_set)
    eer = evaluate(winner_net, eval_set)

    eer_list = evaluate_per_attack(winner_net, eval_set, eval_class_attack)

    print("\n")
    print("**** equal error rate train = {} % ****".format(round(train_eer*100, 1)))

    print("\n")
    print("**** equal error rate dev = {}  ****".format(round(dev_eer*100, 1)))

    print("\n")
    print("**** equal error rate = {}  ****".format(round(eer*100, 1)))

    print("\n")
    for i in range(len(eval_class_attack)):
        print("attack", eval_class_attack[i], round(eer_list[i]*100, 1), "%")

    if separate_known_unknown:
        test_seen_classes = []
        test_unseen_classes = []

        for x in eval_set:
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
