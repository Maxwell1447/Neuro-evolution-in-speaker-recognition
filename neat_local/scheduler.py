from neat.reporting import BaseReporter
from math import cos, pi

class ExponentialScheduler(BaseReporter):
    """
    Reports testing accuracy of best genome every 100 steps.
    Reports complexity of best genome (number of connections) every step.
    """

    def __init__(self, semi_gen=500, final_values=None):
        """
        semi_gen: number of generations required to be half way to th final value
        """

        if final_values is None:
            final_values = {}
        self.final_values = final_values
        self.fade_factor = 0.5 ** (1 / semi_gen)

    def end_generation(self, conf, population, species_set):

        print()
        print("--Scheduler Values--")
        for key in self.final_values:
            prev_value = getattr(conf.genome_config, key)

            print(key, prev_value)

            setattr(conf.genome_config, key,
                    (prev_value - self.final_values[key]) * self.fade_factor
                    + self.final_values[key])


class SineScheduler(BaseReporter):
    """
    Reports testing accuracy of best genome every 100 steps.
    Reports complexity of best genome (number of connections) every step.
    """

    def __init__(self, conf, period=500, final_values=None):
        """
        semi_gen: number of generations required to be half way to th final value
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

        print()
        print("--Scheduler Values--")
        for key in self.final_values:
            b, a = self.final_values[key], self.init_values[key]
            value = (a + b) / 2 + (a - b) / 2 * cos(2 * pi * self.gen / self.period)
            print(key, value)

            setattr(conf.genome_config, key, value)

            self.gen += 1
