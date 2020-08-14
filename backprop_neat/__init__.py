"""A NEAT (NeuroEvolution of Augmenting Topologies) implementation"""
import backprop_neat.nn as nn


from backprop_neat.config import Config
from backprop_neat.population import Population, CompleteExtinctionException
from backprop_neat.genome import DefaultGenome
from backprop_neat.reproduction import DefaultReproduction
from backprop_neat.stagnation import DefaultStagnation
from backprop_neat.reporting import StdOutReporter
from backprop_neat.species import DefaultSpeciesSet
from backprop_neat.statistics import StatisticsReporter
from backprop_neat.parallel import ParallelEvaluator
from backprop_neat.threaded import ThreadedEvaluator
