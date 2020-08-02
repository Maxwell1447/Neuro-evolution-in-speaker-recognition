import neat
import numpy as np
from tqdm import tqdm
import os
import pickle

from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.data_utils_short import ASVDatasetshort
from anti_spoofing.utils_ASV import whiten, gate_mfcc
from anti_spoofing.metrics_utils import rocch2eer, rocch


def evaluate(net, data_loader):
    """
    compute the eer equal error rate
    :param net: network
    :param data_loader: test dataset, contains audio files in a numpy array format
    :return eer
    """
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
        for i in range(6):
            net[i].reset()
        sample_input, output = data[0], data[1]
        sample_input = whiten(sample_input)
        xo = np.zeros(6)
        for i in range(1, 6):
            xo[i] = gate_mfcc(net[i], sample_input)
        if output == 1:
            target_scores_sum.append(xo.sum()/6)
            target_scores_prod.append(xo.prod())
            target_scores_min.append(xo.min())
            target_scores_max.append(xo.max())
            target_scores_median.append(np.median(xo))
        else:
            non_target_scores_sum.append(xo.sum()/6)
            non_target_scores_prod.append(xo.prod())
            non_target_scores_min.append(xo.min())
            non_target_scores_max.append(xo.max())
            non_target_scores_median.append(np.median(xo))

    target_scores_sum = np.array(target_scores_sum)
    non_target_scores_sum = np.array(non_target_scores_sum)

    target_scores_prod = np.array(target_scores_prod)
    non_target_scores_prod = np.array(non_target_scores_prod)

    target_scores_min = np.array(target_scores_min)
    non_target_scores_min = np.array(non_target_scores_min)

    target_scores_max = np.array(target_scores_max)
    non_target_scores_max = np.array(non_target_scores_max)

    target_scores_median = np.array(target_scores_median)
    non_target_scores_median = np.array(non_target_scores_median)

    pmiss_sum, pfa_sum = rocch(target_scores_sum, non_target_scores_sum)
    eer_sum = rocch2eer(pmiss_sum, pfa_sum)

    pmiss_prod, pfa_prod = rocch(target_scores_prod, non_target_scores_prod)
    eer_prod = rocch2eer(pmiss_prod, pfa_prod)

    pmiss_min, pfa_min = rocch(target_scores_min, non_target_scores_min)
    eer_min = rocch2eer(pmiss_min, pfa_min)

    pmiss_max, pfa_max = rocch(target_scores_max, non_target_scores_max)
    eer_max = rocch2eer(pmiss_max, pfa_max)

    pmiss_median, pfa_median = rocch(target_scores_median, non_target_scores_median)
    eer_median = rocch2eer(pmiss_median, pfa_median)

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

    net_1 = pickle.load(open('best_genome_eoc_class_1', 'rb'))
    net_2 = pickle.load(open('best_genome_eoc_class_2', 'rb'))
    net_3 = pickle.load(open('best_genome_eoc_class_3', 'rb'))
    net_4 = pickle.load(open('best_genome_eoc_class_4', 'rb'))
    net_5 = pickle.load(open('best_genome_eoc_class_5', 'rb'))
    net_6 = pickle.load(open('best_genome_eoc_class_6', 'rb'))

    net = [net_1, net_2, net_3, net_4, net_5, net_6]

    trainset = ASVDatasetshort(None, nb_samples=2538, do_mfcc=True)
    devset = ASVDataset(is_train=False, is_eval=False, nb_samples=80000, do_mfcc=True)
    testset = ASVDataset(is_train=False, is_eval=True, nb_samples=80000, do_mfcc=True)

    aggregate_net = []
    for i in range(6):
        aggregate_net.append(neat.nn.RecurrentNetwork.create(net[i], config))

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

    train_eer = evaluate(aggregate_net, trainset)
    dev_eer = evaluate(aggregate_net, devset)
    eer_seen = evaluate(aggregate_net, test_seen_classes)
    eer_unseen = evaluate(aggregate_net, test_unseen_classes)
    eer = evaluate(aggregate_net, testset)

    print("\n")
    print("**** equal error rate train = {}  ****".format(train_eer))

    print("\n")
    print("**** equal error rate dev = {}  ****".format(dev_eer))

    print("\n")
    print("**** equal error rate seen classes = {}  ****".format(eer_seen))

    print("\n")
    print("**** equal error rate unseen classes = {}  ****".format(eer_unseen))

    print("\n")
    print("**** equal error rate = {}  ****".format(eer))
