"""A NEAT (NeuroEvolution of Augmenting Topologies) implementation"""
import backprop_neat.nn as nn


from backprop_neat.config import Config
from backprop_neat.population import Population, CompleteExtinctionException
from backprop_neat.genome import DefaultGenome
from backprop_neat.reproduction import DefaultReproduction
from backprop_neat.stagnation import DefaultStagnation
from neat.reporting import StdOutReporter
from neat.species import DefaultSpeciesSet
from neat.statistics import StatisticsReporter
from neat.parallel import ParallelEvaluator
from neat.threaded import ThreadedEvaluator
