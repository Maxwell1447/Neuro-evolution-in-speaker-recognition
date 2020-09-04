# Backprop NEAT
This is the backprop NEAT package.

## Table of contents
* [General info](#general-info)
* [Note](#note)
* [Future work](#future-work)

## General info
This package is an implementation deeply inspired from NEAT-Python but with one main tweak: a backprop step takes place after the evaluation.

To use backprop NEAT, just import backprop_neat as neat. Besides backpropagation, this package provides several features such as enhanced checkpointers and memory-friendly StatisticsReporters.


## Note
Backprop NEAT should not be used with recurrent networks although there are no implementation of feed forward phenotypes. Instead, you can use RecurrentNet with feed-forward genomes.
This is not a true feed-forward, but it gives almost similar results nonetheless.
The reason why recurrent genomes are incompatible with backpropagation is probably due to vanishing/exploding gradient.

## Future Work

Implement a true feed forward phenotype.
