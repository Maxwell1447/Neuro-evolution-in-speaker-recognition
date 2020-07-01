from __future__ import print_function

import multiprocessing
import os
import neat
from sklearn import datasets
import numpy as np
import random
from sklearn.model_selection import train_test_split
import optuna


"""
NEAT APPLIED TO IRIS DATASET
"""


class NumGenReporter(neat.reporting.BaseReporter):

    def __init__(self):
        super().__init__()
        self.n = 0
        self.fail = True

    def found_solution(self, config, generation, best):
        self.fail = False

    def end_generation(self, config, population, species_set):
        print("-", end='')
        self.n += 1


class IRISMultiEvaluator(neat.ParallelEvaluator):

    def __init__(self, num_workers, eval_function, inputs, outputs):
        super().__init__(num_workers, eval_function)
        self.inputs = inputs
        self.outputs = outputs

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config, self.inputs, self.outputs)))

        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness = job.get(timeout=self.timeout)


def load_iris(features, wrong_labelling=0):
    """
    :param features: indices we choose to keep
    :param wrong_labelling: number of point randomly labeled
    :return: the data train-test split, the inputs and outputs string names (eg. "sepal length (cm)", "versicolor"
    """
    iris = datasets.load_iris()
    iris_x = iris['data'][:, features]
    iris_target = iris['target']

    X_train, X_test, y_train, y_test = train_test_split(iris_x, iris_target, test_size=0.33, random_state=42)

    # wrong labelling
    perturbation_idx = np.random.choice(y_train.shape[0], wrong_labelling)
    y_train[perturbation_idx] = np.random.randint(3, size=wrong_labelling)

    y_train_reformated = []
    for i in range(3):
        y_train_reformated.append((y_train == i).astype(np.float32))
    y_train_reformated = np.array(y_train_reformated).T

    y_test_reformated = []
    for i in range(3):
        y_test_reformated.append((y_test == i).astype(np.float32))
    y_test_reformated = np.array(y_test_reformated).T



    labels_ = [iris['feature_names'][i] for i in features]
    names_ = iris['target_names']

    data = {"X_train": X_train,
            "X_test": X_test,
            "y_train": y_train_reformated,
            "y_test": y_test_reformated}
    return data, labels_, names_


def eval_genomes(genomes, config):
    """
    Most important part of NEAT since it is here that we adapt NEAT to our problem.
    We tell what is the phenotype of a genome and how to calculate its fitness (same idea than a loss)

    :param genomes: list of all the genomes to get evaluated
    :param config: config from the config file
    """
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config, inputs, outputs)


def eval_genome(genome, config, inputs, outputs):

    fitness = 30.

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for xi, xo in zip(inputs, outputs):
        output = np.array(net.activate(xi))
        # update the fitness of each individual:
        # fitness = 30 - MSE_all_data
        fitness -= np.sum((output - xo) ** 2)

    return fitness


def run(config, n_gen=200, params=None):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genontype (winner), the configs, the stats of the run and the accuracy on the testing set
    """

    if params is None:
        params = {}
    for p in params:
        setattr(config.genome_config, p, params[p])

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    num_gen_reporter = NumGenReporter()
    p.add_reporter(num_gen_reporter)

    evaluator = IRISMultiEvaluator(multiprocessing.cpu_count(), eval_genome, inputs, outputs)

    # Run for up to n_gen generations.
    winner = p.run(evaluator.evaluate, n_gen)

    if num_gen_reporter.fail:
        num_gen = n_gen + 1
    else:
        num_gen = num_gen_reporter.n

    return winner, num_gen


def objective(trial):

    local_dir = os.path.dirname(__file__)

    config_path = os.path.join(local_dir, 'config_iris')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    conn_add_prob = trial.suggest_uniform("conn_add_prob", 0.001, 0.6)
    conn_delete_prob = trial.suggest_uniform("conn_delete_prob", 0.001, 0.6)

    node_add_prob = trial.suggest_uniform("node_add_prob", 0.001, 0.6)
    node_delete_prob = trial.suggest_uniform("node_delete_prob", 0.001, 0.6)

    bias_mutate_power = trial.suggest_loguniform("bias_mutate_power", 0.001, 10.)  # 0.01, 1.
    weight_mutate_power = trial.suggest_loguniform("weight_mutate_power", 0.001, 10.)  # 0.01, 1.

    # compatibility_disjoint_coefficient = trial.suggest_uniform("compatibility_disjoint_coefficient", 0.5, 1.2)  # 0.8, 1.2
    # compatibility_weight_coefficient = trial.suggest_uniform("compatibility_weight_coefficient", 0.3, 0.8)  # 0.3, 0.7

    single_structural_mutation = trial.suggest_categorical('single_structural_mutation', [True, False])
    num_hidden = trial.suggest_int("num_hidden", 0, 0)

    # conn_add_prob = 0.01
    # conn_delete_prob = 0.03
    # node_add_prob = 0.066
    # node_delete_prob = 0.05
    # bias_mutate_power = 0.6
    # weight_mutate_power = 0.5
    compatibility_disjoint_coefficient = 1.
    compatibility_weight_coefficient = 0.5

    setattr(config, "pop_size", trial.suggest_int("pop_size", 20, 100))

    params = {"conn_add_prob": conn_add_prob,
              "conn_delete_prob": conn_delete_prob,
              "node_add_prob": node_add_prob,
              "node_delete_prob": node_delete_prob,
              "bias_mutate_power": bias_mutate_power,
              "weight_mutate_power": weight_mutate_power,
              "compatibility_disjoint_coefficient": compatibility_disjoint_coefficient,
              "compatibility_weight_coefficient": compatibility_weight_coefficient,
              "single_structural_mutation": single_structural_mutation,
              "num_hidden": num_hidden
              }

    winner, num_gen = run(config, params=params)
    print()

    return num_gen


if __name__ == '__main__':

    features = [0, 2]         # features indices we choose to keep (subset of [0, 1, 2, 3])
    data, labels, names = load_iris(features, wrong_labelling=0)
    inputs, outputs = data["X_train"], data['y_train']

    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_iris')

    random.seed(0)

    for i in range(20):
        study = optuna.create_study()
        study.optimize(objective, n_trials=10)

        df = study.trials_dataframe()

        if os.path.isfile(os.path.join(local_dir, 'params_runs.csv')):
            header = False
            mode = 'a'
        else:
            header = True
            mode = 'w'
        df.to_csv("params_runs.csv", index=False, mode=mode, header=header)
