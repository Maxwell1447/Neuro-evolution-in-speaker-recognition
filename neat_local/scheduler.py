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

    def end_generation(self, conf, population, species_set):
        """
        Updates the parameters.
        """

        if verbose:
            print()
            print("--Scheduler Values--")
        for key in self.final_values:
            prev_value = getattr(conf.genome_config, key)

            if verbose:
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

        for key in self.final_values:
            self.init_values[key] = getattr(conf.genome_config, key)

    def end_generation(self, conf, population, species_set):
        """
        Updates the parameters.
        """

        if verbose:
            print()
            print("--Scheduler Values--")

        for key in self.final_values:
            b, a = self.final_values[key], self.init_values[key]
            value = (a + b) / 2 + (a - b) / 2 * cos(2 * pi * self.gen / self.period)

            if verbose:
                print(key, value)

            setattr(conf.genome_config, key, value)

            self.gen += 1
