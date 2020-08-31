import neat
import numpy as np
from tqdm import tqdm
import os
import pickle

from anti_spoofing.data_utils import ASVDataset, ASVFile
from anti_spoofing.utils_ASV import whiten, gate_lfcc
from anti_spoofing.metrics_utils import rocch2eer, rocch


def compute_eer(target_scores, non_target_scores):
    """
    Return the equal error rate from the scores
    :param target_scores: numpy array scores of the bonafide files
    :param non_target_scores: numpy array scores of the spoofed files
    :return: eer equal error rate
    """
    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)
    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)
    return eer


def evaluate_different_dataset(net, data_loader):
    """
    compute the eer equal error rate
    :param net: list of networks
    :param data_loader: list of dataset, contains audio files or extracted features in a numpy array format
    :return eer
    """
    nb_net = len(net)
    target_scores_sum = []
    non_target_scores_sum = []
    target_scores_prod = []
    non_target_scores_prod = []
    target_scores_min = []
    non_target_scores_min = []
    target_scores_max = []
    non_target_scores_max = []
    target_scores_median = []
    non_target_scores_median = []
    for index_data in tqdm(range(len(data_loader[0]))):
        xo = np.zeros(nb_net)
        for i in range(nb_net):
            net[i].reset()
            sample_input, output = data_loader[i][index_data][0], data_loader[i][index_data][1]
            # sample_input = whiten(sample_input)
            xo[i] = gate_lfcc(net[i], sample_input)
        if output == 1:
            target_scores_sum.append(xo.mean())
            target_scores_prod.append(xo.prod())
            target_scores_min.append(xo.min())
            target_scores_max.append(xo.max())
            target_scores_median.append(np.median(xo))
        else:
            non_target_scores_sum.append(xo.mean())
            non_target_scores_prod.append(xo.prod())
            non_target_scores_min.append(xo.min())
            non_target_scores_max.append(xo.max())
            non_target_scores_median.append(np.median(xo))

    eer_sum = compute_eer(target_scores_sum, non_target_scores_sum)
    eer_prod = compute_eer(target_scores_prod, non_target_scores_prod)
    eer_min = compute_eer(target_scores_min, non_target_scores_min)
    eer_max = compute_eer(target_scores_max, non_target_scores_max)
    eer_median = compute_eer(target_scores_median, non_target_scores_median)

    return eer_sum, eer_prod, eer_min, eer_max, eer_median


def evaluate(net, data_loader):
    """
    compute the eer equal error rate
    :param net: list of networks
    :param data_loader: dataset, contains audio files or extracted features in a numpy array format
    :return eer
    """
    nb_net = len(net)
    target_scores_sum = []
    non_target_scores_sum = []
    target_scores_prod = []
    non_target_scores_prod = []
    target_scores_min = []
    non_target_scores_min = []
    target_scores_max = []
    non_target_scores_max = []
    target_scores_median = []
    non_target_scores_median = []
    for data in tqdm(data_loader):
        for i in range(nb_net):
            net[i].reset()
        sample_input, output = data[0], data[1]
        # sample_input = whiten(sample_input)
        xo = np.zeros(nb_net)
        for i in range(nb_net):
            xo[i] = gate_lfcc(net[i], sample_input)
        if output == 1:
            target_scores_sum.append(xo.mean())
            target_scores_prod.append(xo.prod())
            target_scores_min.append(xo.min())
            target_scores_max.append(xo.max())
            target_scores_median.append(np.median(xo))
        else:
            non_target_scores_sum.append(xo.mean())
            non_target_scores_prod.append(xo.prod())
            non_target_scores_min.append(xo.min())
            non_target_scores_max.append(xo.max())
            non_target_scores_median.append(np.median(xo))

    eer_sum = compute_eer(target_scores_sum, non_target_scores_sum)
    eer_prod = compute_eer(target_scores_prod, non_target_scores_prod)
    eer_min = compute_eer(target_scores_min, non_target_scores_min)
    eer_max = compute_eer(target_scores_max, non_target_scores_max)
    eer_median = compute_eer(target_scores_median, non_target_scores_median)

    return eer_sum, eer_prod, eer_min, eer_max, eer_median


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    net_1 = pickle.load(open('best_genome_eoc_class_1_lfcc', 'rb'))
    net_2 = pickle.load(open('best_genome_eoc_class_2_lfcc', 'rb'))
    net_3 = pickle.load(open('best_genome_eoc_class_3_lfcc', 'rb'))
    net_4 = pickle.load(open('best_genome_eoc_class_4_lfcc', 'rb'))
    net_5 = pickle.load(open('best_genome_eoc_class_5_lfcc', 'rb'))
    net_6 = pickle.load(open('best_genome_eoc_class_6_lfcc', 'rb'))

    net = [net_1, net_2, net_3, net_4, net_5, net_6]

    aggregate_net = []
    for i in range(len(net)):
        aggregate_net.append(neat.nn.RecurrentNetwork.create(net[i], config))

    train_lfcc = ASVDataset(is_train=True, is_eval=False, do_lfcc=True, nb_samples=80000, do_standardize=True)
    dev_lfcc = ASVDataset(is_train=False, is_eval=False, do_lfcc=True, nb_samples=80000, do_standardize=True)
    eval_lfcc = ASVDataset(is_train=False, is_eval=True, do_lfcc=True, nb_samples=80000, do_standardize=True)

    for i in range(len(aggregate_net)):
        train_eer = evaluate([aggregate_net[i]], train_lfcc)
        dev_eer = evaluate([aggregate_net[i]], dev_lfcc)
        eer = evaluate([aggregate_net[i]], eval_lfcc)

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
    eer_seen = evaluate(aggregate_net, test_seen_classes)
    eer_unseen = evaluate(aggregate_net, test_unseen_classes)
    
    print("\n")
    print("**** equal error rate seen classes = {}  ****".format(eer_seen))

    print("\n")
    print("**** equal error rate unseen classes = {}  ****".format(eer_unseen))
    """