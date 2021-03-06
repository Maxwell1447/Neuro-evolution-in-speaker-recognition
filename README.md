# Neuro evolution applied to speaker recognition
The goal of this project is to apply NEAT rather than usual deep learning models to speaker recognition and anti-spoofing 
to tackle the inconveniences of a large pre-trained model on embedded systems.
We used libraries available in python, [NEAT-Python](https://github.com/neat-python/neat-python) 
and [Pytorch-NEAT](https://github.com/uber-research/PyTorch-NEAT),
to deal with the following databases:
- Iris
- MNIST
- audio gender classification with the Libri speech (helped with [this repository](https://github.com/oscarknagg/raw-audio-gender-classification))
- Anti-spoofing with ASVSpoof2019

We tried to understand the ability of NEAT and its variants to deal with classification tasks. Our main objective was to achieve the best performance in ASVSpoof2019.

## Table of contents
* [Context](#context)
* [General info](#general-info)
* [Setup](#setup)
* [Aspect covered](#aspect-covered)
* [Status](#status)
* [Inspiration](#references)
* [Authors](#authors)

## Context

All the work has been performed by students in Data Science from Eurecom. This was in the context of the **Automatic Speaker Verification
Spoofing And Countermeasures Challenge** started in 2019. The students worked in a team of researcher from Eurecom's Digital Security Department.

### **Students**

- Arnaud Barral

- Maxime Bouthors

### **Supervisors**

- [Nicholas Evans](https://www.eurecom.fr/~evans)

- [Massimiliano Todisco](https://www.massimilianotodisco.eu)

- Jose Patino


## General info
Neuroevolution, or neuro-evolution, uses evolutionary algorithms to generate artificial neural networks (ANN), 
parameters, topology and rules. It can be contrasted with conventional deep learning techniques 
that use gradient descent on a neural network with a fixed topology. 
The neuroevolution algorithm that is used in this project is NEAT (Neuroevolution of Augmenting Topology), a popular method that aims 
to give ANN of minimalistic sizes. For further information about NEAT and its algorithm, please consult the 
[user's page](https://www.cs.ucf.edu/~kstanley/neat.html) associated.

We are two students from EURECOM ( http://www.eurecom.fr/en ) both following the data science track.
This repository contains our work from our project on Neuro evolution. 
The aim for this project is to apply neuro evolution to speaker recognition.

We covered various aspects of NEAT. You can find a README file for every folder in this project explaining 
the aspect treated, and how to run the code.

## Setup

Run ```pip install -r requirement.txt``` to install the required packages.

To generate the graphics of the topologies of the genomes, you need to install [graphviz](https://www.graphviz.org/download/).
Make sure that the directory containing the *dot* executable is on your system’s path.

## Aspects covered
List of aspects covered:
* [NEAT in classification tasks (IRIS)](iris)
* [NEAT in audio gender classification tasks](raw_audio_gender_classification)
* [GPU vs CPU in NEAT evaluation](gpu_tests)
* [CNN-HyperNEAT applied on MNIST](hyperneat)
* [NEAT applied on anti-spoofing](anti_spoofing)
* [Backprop NEAT](backprop_neat)


## Status
Project finished. Some modifications may still occur though.

## References

[Valenti, Giacomo and  Delgado, Héctor and  Todisco, Massimiliano and  Evans, Nicholas and  Pilati, Laurent, 
(2018),
*An end-to-end spoofing countermeasure for automatic speaker verification using evolving recurrent neural networks*
](http://www.eurecom.fr/fr/publication/5523/detail/an-end-to-end-spoofing-countermeasure-for-automatic-speaker-verification-using-evolving-recurrent-neural-networks)

## Authors
Created by Maxime BOUTHORS and Arnaud BARRAL
