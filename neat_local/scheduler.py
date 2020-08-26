from neat.reporting import BaseReporter
from math import cos, pi, exp
import numpy as np
import matplotlib.pyplot as plt


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
        self.verbose = verbose

    def post_evaluate(self, conf, population, species, best_genome):
        """
        Updates the parameters.
        """

        if self.verbose:
            print()
            print("--Exp Scheduler Values--")
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

    def __init__(self, conf, period=500, final_values=None, verbose=0, monitor=True):
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

        if monitor:
            self.values = {}
            for key in self.final_values:
                self.values[key] = []

        self.monitor = monitor

    def post_evaluate(self, config, population, species, best_genome):
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

            setattr(config.genome_config, key, value)

            if self.monitor:
                self.values[key].append(value)
        self.gen += 1

    def display(self):

        assert self.monitor

        plt.figure()
        plt.xlabel("generations")
        plt.ylabel("parameter values")
        plt.title("Parameter scheduling")
        for key in self.final_values:
            plt.plot(self.values[key], label=key)
        plt.show()


class SquashedSineScheduler(BaseReporter):
    """
    Scheduler with "squashed" sine variation over time: the parameters will oscillate between
    there initial value and the corresponding value in 'final_value' with a period of
    'period'.
    The idea is that higher values of 'learning rate' are associated with exploratory phases
    whereas lower values are associated to pruning and fine-tuning of the weights+biases.
    The 'learning rate' is decomposed for NEAT:
        * for the topology: node_add_prob and conn_add_prob
        * for the weights/biases: weight_mutate_power and bias_mutate_power
    """

    def __init__(self, conf, period=500, final_values=None, alpha=1, verbose=0, monitor=True, offset=0):
        """
        period: number of generation required to loop
        verbose: if 1, print the current values of the parameters
        conf: NEAT config
        alpha: squash factor of the sine. Greater alpha squashes more the sine.
        """

        if final_values is None:
            final_values = {}
        self.final_values = final_values
        self.period = period
        self.gen = 0
        self.init_values = {}
        self.verbose = verbose
        self.alpha = alpha
        self.offset = offset

        for key in self.final_values:
            self.init_values[key] = getattr(conf.genome_config, key)

        if monitor:
            self.values = {}
            for key in self.final_values:
                self.values[key] = []

        self.monitor = monitor

    def post_evaluate(self, config, population, species, best_genome):
        """
        Updates the parameters.
        """

        if self.verbose:
            print()
            print("--Squashed Sine Scheduler Values--")

        for key in self.final_values:
            a, b = self.final_values[key], self.init_values[key]
            co = cos(2 * pi * (self.gen - self.offset) / self.period)
            value = a + (b - a) * (exp(self.alpha * (co + 1)/2) - 1) / (exp(self.alpha) + 1)

            if self.verbose:
                print(key, value)

            setattr(config.genome_config, key, value)

            if self.monitor:
                self.values[key].append(value)
        self.gen += 1

    def display(self):

        assert self.monitor

        plt.figure()
        plt.xlabel("generations")
        plt.ylabel("parameter values")
        plt.title("Parameter scheduling")
        for key in self.final_values:
            plt.plot(self.values[key], label=key)
        plt.show()


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


class MutateScheduler(BaseReporter):
    """
    Scheduler that mutates the learning rate when on a plateau
    The 'learning rate' is decomposed for NEAT:
        * for the topology: node_add_prob and conn_add_prob
        * for the weights/biases: weight_mutate_power and bias_mutate_power
    """

    def __init__(self, parameters=None, verbose=0, patience=5, momentum=0.99, monitor=True):
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
        self.patience = patience
        self.cpt = 0
        self.mu = momentum
        self.w = 1.
        self.s = 0.
        self.monitor = monitor

        if monitor:
            self.values = {}
            for key in parameters:
                self.values[key] = []

    def post_evaluate(self, config, population, species, best_genome):
        v = best_genome.fitness

        self.s = self.s * self.mu + v
        smoothed_fitness = self.s / self.w
        self.w = 1 + self.mu * self.w

        if self.last_fitness is not None:

            if smoothed_fitness <= self.last_fitness:
                self.cpt += 1
                if self.cpt > self.patience:

                    factor = np.exp(np.random.randn() / 1000)
                    for key in self.parameters:
                        setattr(config.genome_config, key, factor * getattr(config.genome_config, key))
                    self.cpt = 0
                    self.last_fitness = smoothed_fitness
            else:
                self.last_fitness = smoothed_fitness
                self.cpt = 0

        if self.monitor:
            for key in self.parameters:
                self.values[key].append(getattr(config.genome_config, key))

        else:
            self.last_fitness = smoothed_fitness

        if self.verbose:
            for key in self.parameters:
                print(key, getattr(config.genome_config, key))

    def display(self):

        assert self.monitor

        plt.figure()
        plt.xlabel("generations")
        plt.ylabel("parameter values")
        plt.title("Parameter scheduling")
        for key in self.parameters:
            plt.plot(self.values[key], label=key)
        plt.show()


class ImpulseScheduler(BaseReporter):
    """
    Scheduler that creates an impulse in the learning rate when on a plateau
    The 'learning rate' is decomposed for NEAT:
        * for the topology: node_add_prob and conn_add_prob
        * for the weights/biases: weight_mutate_power and bias_mutate_power
    """

    def __init__(self, parameters=None, verbose=0, patience=5, momentum=0.99, impulse_duration=10, impulse_factor=2,
                 monitor=True):
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
        self.patience = patience
        self.cpt = 0
        self.mu = momentum
        self.w = 1.
        self.s = 0.
        self.impulse_cpt = 0
        self.impulse_duration = impulse_duration
        self.impulse_factor = impulse_factor
        self.monitor = monitor
        if monitor:
            self.values = {}
            for key in self.parameters:
                self.values[key] = []

    def post_evaluate(self, config, population, species, best_genome):
        v = best_genome.fitness

        self.s = self.s * self.mu + v
        smoothed_fitness = self.s / self.w
        self.w = 1 + self.mu * self.w

        if self.last_fitness is not None:

            if smoothed_fitness <= self.last_fitness:

                if self.impulse_cpt > 0:
                    self.impulse_cpt -= 1
                    if self.impulse_cpt == 0:
                        for key in self.parameters:
                            setattr(config.genome_config, key,
                                    getattr(config.genome_config, key) / self.impulse_factor)
                    self.last_fitness = smoothed_fitness
                else:
                    self.cpt += 1
                    if self.cpt > self.patience:
                        for key in self.parameters:
                            setattr(config.genome_config, key,
                                    self.impulse_factor * getattr(config.genome_config, key))
                        self.cpt = 0
                        self.last_fitness = smoothed_fitness
                        self.impulse_cpt = self.impulse_duration
            else:
                self.last_fitness = smoothed_fitness
                self.cpt = 0

        else:
            self.last_fitness = smoothed_fitness

        if self.verbose:
            for key in self.parameters:
                print(key, getattr(config.genome_config, key))

        if self.monitor:
            for key in self.parameters:
                self.values[key].append(getattr(config.genome_config, key))

    def display(self):

        assert self.monitor

        plt.figure()
        plt.xlabel("generations")
        plt.ylabel("parameter values")
        plt.title("Parameter scheduling")
        for key in self.parameters:
            plt.plot(self.values[key], label=key)
        plt.show()


class BackpropScheduler(BaseReporter):
    """
    """

    def __init__(self, config, patience=100):
        """
        """
        setattr(config.genome_config, "backprop", False)
        self.patience = patience
        self.cpt = 0

    def end_generation(self, config, population, species_set):

        if self.cpt == self.patience:
            setattr(config.genome_config, "backprop", True)
        else:
            self.cpt += 1


class CyclicBackpropScheduler(BaseReporter):
    """
    """

    def __init__(self, config, patience=100, period=200, offset=50, start=-1, monitor=False):
        """
        """
        setattr(config.genome_config, "backprop", False)
        self.patience = patience
        self.offset = offset
        self.cpt = 0
        self.period = period
        self.start = start
        self.monitor = monitor
        if monitor:
            self.value = []

    def end_generation(self, config, population, species_set):

        if self.cpt > self.start:
            if self.cpt % self.period == self.patience - self.offset:
                setattr(config.genome_config, "backprop", True)

            if self.cpt % self.period == -self.offset % self.period:
                setattr(config.genome_config, "backprop", False)

        if self.cpt == self.start:
            setattr(config.genome_config, "backprop", True)

        self.cpt += 1
        if self.monitor:
            self.value.append(getattr(config.genome_config, "backprop"))

    def display(self):

        assert self.monitor

        plt.figure()
        plt.xlabel("generations")
        plt.ylabel("backprop")
        plt.title("Backprop scheduling")
        plt.plot(self.value)
        plt.show()


class EarlyExplorationScheduler(BaseReporter):

    def __init__(self, conf, duration=20, values=None, verbose=0, reset=False, monitor=False):
        """
        period: number of generation required to loop
        verbose: if 1, print the current values of the parameters
        conf: NEAT config
        """

        if values is None:
            values = {}
        self.values = values
        self.initial_values = {}
        self.monitor = monitor
        self.all_values = {}
        if self.monitor:
            for key in values:
                self.all_values[key] = []

        for key in values:
            self.initial_values[key] = getattr(conf.genome_config, key)
        self.duration = duration
        self.gen = 0
        self.verbose = verbose
        self.reset = reset

    def post_evaluate(self, config, population, species, best_genome):
        """
        Updates the parameters.
        """

        if self.gen < self.duration:

            for key in self.values:

                setattr(config.genome_config, key, self.values[key])

            self.gen += 1

        elif self.gen == self.duration:

            if self.reset:
                for key in self.values:
                    setattr(config.genome_config, key, self.initial_values[key])

            self.gen += 1

        if self.monitor:
            if self.monitor:
                for key in self.values:
                    self.all_values[key].append(getattr(config.genome_config, key))

        if self.verbose:
            print()
            print("--Early exploration Scheduler Values--")

            for key in self.values:
                print(key, getattr(config.genome_config, key))

    def display(self):

        assert self.monitor

        for key in self.all_values:
            plt.figure()
            plt.xlabel("generations")
            plt.ylabel("parameter values")
            plt.title("{} scheduling".format(key))

            plt.plot(self.all_values[key], label=key)
            plt.show()


class AdaptiveBackpropScheduler(BaseReporter):

    def __init__(self, conf, patience=20, values=None, semi_gen=50, monitor=False, start=0, patience_before_backprop=50,
                 privilege=50):

        self.init_values = {}
        self.all_values = {}

        if values is None:
            values = []

        for key in values:
            self.init_values[key] = getattr(conf.genome_config, key)
            if monitor:
                self.all_values[key] = []

        self.start = start
        self.gen = 0
        self.cpt = 0
        self.privilege = privilege
        self.patience_before_backprop = patience_before_backprop
        self.fade_factor = 0.5 ** (1 / semi_gen)
        self.backprop = False
        self.momentum = 0.95
        self.s = 0
        self.w = 1
        self.last_fitnesses = [0 for _ in range(patience)]
        self.monitor = monitor
        setattr(conf.genome_config, "backprop", False)

    def post_evaluate(self, config, population, species, best_genome):
        """
        Updates the parameters.
        """

        if self.gen > self.start:

            fitness = best_genome.fitness
            self.s = self.s * self.momentum + fitness
            smoothed_fitness = self.s / self.w
            self.w = 1 + self.momentum * self.w

            for key in self.init_values:
                setattr(config.genome_config, key, getattr(config.genome_config, key) * self.fade_factor)

            if self.backprop:

                if self.cpt > self.patience_before_backprop + self.privilege \
                        and smoothed_fitness <= min(self.last_fitnesses):  # if plateau
                    self.cpt = 0
                    for key in self.init_values:
                        setattr(config.genome_config, key, self.init_values[key])
                    self.backprop = False
                    setattr(config.genome_config, "backprop", False)
            elif self.cpt == self.patience_before_backprop:
                self.backprop = True
                setattr(config.genome_config, "backprop", True)

            self.last_fitnesses.pop(0)
            self.last_fitnesses.append(smoothed_fitness)
            self.cpt += 1

        elif self.gen == self.start:
            self.cpt = 0
            for key in self.init_values:
                setattr(config.genome_config, key, self.init_values[key])
            self.backprop = False
            setattr(config.genome_config, "backprop", False)

        if self.monitor:
            for key in self.init_values:
                self.all_values[key].append(getattr(config.genome_config, key))

        self.gen += 1

    def display(self):

        assert self.monitor

        for key in self.all_values:
            plt.figure()
            plt.xlabel("generations")
            plt.ylabel("parameter values")
            plt.title("{} scheduling".format(key))

            plt.plot(self.all_values[key], label=key)
            plt.show()


class DisableBackpropScheduler(BaseReporter):

    def post_evaluate(self, config, population, species, best_genome):
        """
        Updates the parameters.
        """

        setattr(config.genome_config, "backprop", False)
