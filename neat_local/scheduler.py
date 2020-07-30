from neat.reporting import BaseReporter
from math import cos, pi


class ExponentialScheduler(BaseReporter):
    """
    Scheduler that allows parameter decay.
    The initial values are those put in the config file.
    The asymptotic values are those given in dictionnary 'final_values'.
    Only the parameters described in 'final_values' will be affectd by the sheduler.
    The decay factor is describe by 'semi_gen', the number of generations necessary to be
    half way to the final value.
    """

    def __init__(self, semi_gen=500, final_values=None, verbose=0):
        """
        semi_gen: number of generations required to be half way to th final value
        final_values: dictionnary of format {'parameter_name' (string): final value (float)}
        e.g. final_values = {"node_add_prob": 0.05, "conn_add_prob": 0.02}
        verbose: if 1, print the current values of the parameters
        """

        if final_values is None:
            final_values = {}
        self.final_values = final_values
        self.fade_factor = 0.5 ** (1 / semi_gen)
        self.verbose= verbose

    def end_generation(self, conf, population, species_set):
        """
        Updates the parameters.
        """

        if self.verbose:
            print()
            print("--Scheduler Values--")
        for key in self.final_values:
            prev_value = getattr(conf.genome_config, key)

            if self.verbose:
                print(key, prev_value)

            setattr(conf.genome_config, key,
                    (prev_value - self.final_values[key]) * self.fade_factor
                    + self.final_values[key])


class SineScheduler(BaseReporter):
    """
    Scheduler with sine variation over time: the parameters will oscillate between
    there initial value and the corresponding value in 'final_value' with a period of
    'period'.
    The idea is that higher values of 'learning rate' are associated with exploratory phases
    whereas lower values are associated to pruning and fine-tuning of the weights+biases.
    The 'learning rate' is decomposed for NEAT:
        * for the topology: node_add_prob and conn_add_prob
        * for the weights/biases: weight_mutate_power and bias_mutate_power
    """

    def __init__(self, conf, period=500, final_values=None, verbose=0):
        """
        period: number of generation required to loop
        verbose: if 1, print the current values of the parameters
        conf: NEAT config
        """

        if final_values is None:
            final_values = {}
        self.final_values = final_values
        self.period = period
        self.gen = 0
        self.init_values = {}
        self.verbose = verbose

        for key in self.final_values:
            self.init_values[key] = getattr(conf.genome_config, key)

    def end_generation(self, conf, population, species_set):
        """
        Updates the parameters.
        """

        if self.verbose:
            print()
            print("--Scheduler Values--")

        for key in self.final_values:
            b, a = self.final_values[key], self.init_values[key]
            value = (a + b) / 2 + (a - b) / 2 * cos(2 * pi * self.gen / self.period)

            if self.verbose:
                print(key, value)

            setattr(conf.genome_config, key, value)

            self.gen += 1


class OnPlateauScheduler(BaseReporter):
    """
    Scheduler that either shrinks or amplify the 'learning' rate by a certain factor when on a plateau
    The 'learning rate' is decomposed for NEAT:
        * for the topology: node_add_prob and conn_add_prob
        * for the weights/biases: weight_mutate_power and bias_mutate_power
    """

    def __init__(self, parameters=None, verbose=0, patience=5, factor=0.95, momentum=0.99):
        """
        parameters: name of affected parameters
        verbose: if 1, print the current values of the parameters
        conf: NEAT config
        """

        if parameters is None:
            parameters = []
        self.parameters = parameters
        self.verbose = verbose
        self.last_fitness = None
        self.factor = factor
        self.patience = patience
        self.cpt = 0
        self.mu = momentum
        self.w = 1.
        self.s = 0.

    def post_evaluate(self, config, population, species, best_genome):
        v = best_genome.fitness

        self.s = self.s * self.mu + v
        smoothed_fitness = self.s / self.w
        self.w = 1 + self.mu * self.w

        if self.last_fitness is not None:

            if smoothed_fitness <= self.last_fitness:
                self.cpt += 1
                if self.cpt > self.patience:
                    for key in self.parameters:
                        setattr(config.genome_config, key, self.factor * getattr(config.genome_config, key))
                    self.cpt = 0
                    self.last_fitness = smoothed_fitness
            else:
                self.last_fitness = smoothed_fitness
                self.cpt = 0

        else:
            self.last_fitness = smoothed_fitness

        if self.verbose:
            for key in self.parameters:
                print(key, getattr(config.genome_config, key))
