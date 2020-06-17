# Neuro evolution applied to speaker recognition
The goal of this project is to apply NEAT rather than usual deep learning models to speaker recognition and anti-spoofing 
to tackle the inconveniences of a large pre-trained model on embedded systems.
We used libraries available in python, [NEAT-Python](https://github.com/neat-python/neat-python) 
and [Pytorch-NEAT](https://github.com/uber-research/PyTorch-NEAT),
to deal with the following databases:
- Iris
- MNIST
- audio gender classification with the Libri speech
- Anti spoofing with the ASVspoof 2019

Our goal was to understand the ability of NEAT and its variants to deal with classification tasks.

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Aspect covered](#Aspect covered)
* [Status](#status)
* [Inspiration](#inspiration)
* [Authors](#authors)

## General info
Neuroevolution, or neuro-evolution, uses evolutionary algorithms to generate artificial neural networks (ANN), 
parameters, topology and rules. It can be contrasted with conventional deep learning techniques 
that use gradient descent on a neural network with a fixed topology. 
The neuroevolution algorithm that is used in this project is NEAT (Neuroevolution of Augmenting Topology), a popular method that aims 
to give ANN of minimalistic sizes. 

We are two students from EURECOM ( http://www.eurecom.fr/en ) both following the data science track.
This repository contains our work from our project on Neuro evolution. 
The aim for this project is to apply neuro evolution to speaker recognition.

We covered various aspects of NEAT. You can find a README file for every folder in this project explaining 
the aspect treated, and how to run the code.

## Setup

Run ```pip install -r requirement.txt``` to install the required packages.

To generate the graphics of the topologies of the genomes, you need to install [graphviz](https://www.graphviz.org/download/).
Make sure that the directory containing the *dot* executable is on your systems’ path.

## Aspect covered
List of aspects covered:
* [NEAT in classification tasks (IRIS)](iris)
* [NEAT in audio gender classification tasks](raw_audio_gender_classification)
* [GPU vs CPU in NEAT evaluation](gpu_tests)
* [CNN-HyperNEAT applied on MNIST](hyperneat)
* [NEAT applied on anti-spoofing](anti_spoofing)

To-do list:
* ES-HyperNEAT

## Status
Project is currently in progress

## References
@misc{neat-python,
    Title = {neat-python},
    Author = {Alan McIntyre and Matt Kallada and Cesar G. Miguel and Carolina Feher da Silva},
    howpublished = {\url{https://github.com/CodeReclaimers/neat-python }}   
  }
  
@misc{PyTorch-NEAT,
	Title = {PyTorch NEAT},
    Author = {Alex Gajewsky},
    howpublished = {\url{https://github.com/uber-research/PyTorch-NEAT/ }}   
  }
  
@misc{raw-audio-gender-classification,
	Title = {raw-audio-gender-classification},
    Author = {Oscar Knagg},
    howpublished = {\url{https://github.com/oscarknagg/raw-audio-gender-classification }
		    \url{https://medium.com/@oknagg/gender-classification-from-raw-audio-with-1d-convolutions-969c82e6b3d1 }}   
  }

## Authors
Created by Maxime BOUTHORS and Arnaud BARRAL
