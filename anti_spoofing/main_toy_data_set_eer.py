import os
import neat
import numpy as np
import multiprocessing
from tqdm import tqdm

from anti_spoofing.data_utils import ASVDataset
from anti_spoofing.metrics_utils import rocch2eer, rocch
from anti_spoofing.utils_ASV import whiten, gate_activation, gate_average, make_visualize


"""
NEAT APPLIED TO ASVspoof 2019
"""

nb_samples_train = 10  # number of audio files used for training
nb_samples_test = 10  # number of audio files used for testing

index_train = [k for k in range(5)] + [k for k in range(2590, 2595)]  # index of audio files to use for training

n_processes = multiprocessing.cpu_count()  # number of workers to use for evaluating the fitness
n_generation = 60  # number of generations

train_loader = ASVDataset(None, is_train=True, is_eval=False, index_list=index_train)
test_loader = ASVDataset(None, is_train=False, is_eval=False,  index_list=index_train)


trainloader = []
for data in train_loader:
    inputs, output = data[0], data[1]
    inputs = whiten(inputs)
    trainloader.append((inputs, output))
    
testloader = []
for data in test_loader:
    inputs, output = data[0], data[1]
    inputs = whiten(inputs)
    testloader.append((inputs, output))


def eval_genomes(genomes, config_):
    """
    Most important part of NEAT since it is here that we adapt NEAT to our problem.
    We tell what is the phenotype of a genome and how to calculate its fitness (same idea than a loss)
    :param config_: config from the config file
    :param genomes: list of all the genomes to get evaluated
    """
    for _, genome in tqdm(genomes):
        net = neat.nn.RecurrentNetwork.create(genome, config_)
        target_scores = []
        non_target_scores = []
        for data in trainloader:
            inputs, output = data[0], data[1]
            net.reset()
            mask, score = gate_activation(net, inputs)
            selected_score = score[mask]
            if selected_score.size == 0:
                xo = 0.5
            else:
                xo = np.sum(selected_score) / selected_score.size
            if output == 1:
                target_scores.append(xo)
            else:
                non_target_scores.append(xo)

        target_scores = np.array(target_scores)
        non_target_scores = np.array(non_target_scores)
        
        pmiss, pfa = rocch(target_scores, non_target_scores)
        eer = rocch2eer(pmiss, pfa)
        genome.fitness = 2 * (.5 - eer)
        
        

def eval_genome(genome, config_):
    """
    Most important part of NEAT since it is here that we adapt NEAT to our problem.
    We tell what is the phenotype of a genome and how to calculate its fitness 
    (same idea than a loss)
    :param config_: config from the config file
    :param genome: list of all the genomes to get evaluated
    this version is intented to use ParallelEvaluator and should be much faster
    """
    net = neat.nn.RecurrentNetwork.create(genome, config_)
    target_scores = []
    non_target_scores = []
    for data in trainloader:
        inputs, output = data[0], data[1]
        net.reset()
        """
        score_weight, score = gate_activation(net, inputs)
        selected_score = score[mask]
        selected_score = gate_activation(net, inputs)
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        """
        selected_score = gate_average(net, inputs)
        xo = np.sum(selected_score) / selected_score.size
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)
            
    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)

    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)

    return 2 * (.5 - eer)
        

def evaluate(net, data_loader):

    correct = 0
    total = 0
    net.reset()
    target_scores = []
    non_target_scores = []
    for data in tqdm(data_loader):
        inputs, output = data[0], data[1]
        """
        score_weight, score = gate_activation(net, inputs)
        selected_score = score[mask]
        selected_score = gate_activation(net, inputs)
        if selected_score.size == 0:
            xo = 0.5
        else:
            xo = np.sum(selected_score) / selected_score.size
        """
        selected_score = gate_average(net, inputs)
        xo = np.sum(selected_score) / selected_score.size
        total += 1
        correct += ((xo > 0.5) == output)
        if output == 1:
            target_scores.append(xo)
        else:
            non_target_scores.append(xo)
        
    target_scores = np.array(target_scores)
    non_target_scores = np.array(non_target_scores)
    
    pmiss, pfa = rocch(target_scores, non_target_scores)
    eer = rocch2eer(pmiss, pfa)
    
    return target_scores, non_target_scores, float(correct)/total, eer


def run(config_file, n_gen):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genontype (winner), the configs, the stats of the run and the accuracy on the testing set
    """
    # Load configuration.
    config_ = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config_)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats_ = neat.StatisticsReporter()
    p.add_reporter(stats_)
    # p.add_reporter(neat.Checkpointer(5))

    # Run for up to n_gen generations.
    # multi processing
    if n_processes > 1:
        pe = neat.ParallelEvaluator(n_processes, eval_genome)
        winner_ = p.run(pe.evaluate, n_gen)
    else:
        winner_ = p.run(eval_genomes, n_gen)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner_))

    # Show output of the most fit genome against training data.
    print('\n')
    winner_net = neat.nn.RecurrentNetwork.create(winner_, config_)

    training_target_scores, training_non_target_scores, training_accuracy, training_eer = evaluate(winner_net, trainloader)
    target_scores, non_target_scores, accuracy, eer = evaluate(winner_net, testloader)

    print("**** training accuracy = {}  ****".format(training_accuracy))
    print("**** training target scores = {}  ****".format(training_target_scores))
    print("**** training non target scores = {}  ****".format(training_non_target_scores))
    print("**** training equal error rate = {}  ****".format(training_eer))


    print("\n")
    print("**** accuracy = {}  ****".format(accuracy))
    print("**** testing target scores = {}  ****".format(target_scores))
    print("**** testing non target scores = {}  ****".format(non_target_scores))
    print("**** equal error rate = {}  ****".format(eer))


    return winner_, config_, stats_


if __name__ == '__main__':

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat.cfg')

    winner, config, stats = run(config_path, n_generation)
    make_visualize(winner, config, stats)