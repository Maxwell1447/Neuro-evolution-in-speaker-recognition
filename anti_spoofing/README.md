# Anti spoofing
This is the anti spoofing folder. Here are the scripts for running neat on the ASVspoof 2019 database.

## Table of contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Files description](#files-description)
* [Note](#note)
* [Future work](#future-work)

## General info
We are using the ASVspoof 2019 logical (LA) database.
The logical train audio files are used for training.
The logical dev and eval audio files are used for testing.

We have tested several fitness functions, the mean square error (mse), 
the cross entropy (ce), the equal error rate (eer),
the ease of classification (eoc) and variant of ease of classification.

We use numpy arrays and tensors.

## Screenshots
![Example screenshot](./img/Digraph.jpg)

Topology of a genome obtained by running neat on the ASVspoof 2019 database.


## Code Examples

Loading the dataset, for more details please look at the data_utils files.
```python
train_loader = ASVDatasetshort(None, nb_samples=nb_samples_train)
```

run function present in every main_(...).py files in order to run neat:
```python
def run(config_file, n_gen):
    """
    Launches a run until convergence or max number of generation reached
    :param config_file: path to the config file
    :param n_gen: lax number of generation
    :return: the best genontype (winner), the configs, the stats of the run and the accuracy on the testing set
    """
    # Load configuration.
    config_ = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
                          config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config_)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats_ = neat.StatisticsReporter()
    p.add_reporter(stats_)
    p.add_reporter(neat.Checkpointer(generation_interval=100))

    # Run for up to n_gen generations.
    multi_evaluator = Anti_spoofing_Evaluator(n_processes, eval_genome, batch_size, train_loader)
    winner_ = p.run(multi_evaluator.evaluate, n_gen)
```

## Files description

* Python files for running neat are named like the following:
    * main_\[ dataset for training ]\_[ fitness function ]\.py
     
     * For example: main_toy_data_set_mse means that
        * The dataset used is the toy dataset
        * The fitness function used is the mean square error
     
     * Regarding the dataset:
        * train means that we are using the entire logical train dataset for training
            * It is only used with neat-pytorch since it is much faster
        * train_short means that we are using a subset of the logical train dataset for training
            * It is used with neat-python
        * toy_data_set means that we are using 10 files from the logical train dataset for training
            * * It is used with neat-python

    *  Inside the main files, you can find: 
        * run function (always present)
        * eval genome(s) for computing the fitness
            * eval_genome is used for single processing
            * eval_genomes is used for multi processing
            * for neat-pytorch implementation (main_train(_fitness_funtion)), 
            see the corresponding eval_functions file
     
* utils.py contains auxiliary code to normalize audio files, use gates, ...

* data\_utils.py and data\_utils\_short.py files are used for loading the dataset.
    * Adapted from https://github.com/nesl/asvspoof2019, Author: Moustafa Alzantot
    * data_utils_visualization.py and data_utils_short_visualization.py files plot some 
    statistics about the dataset (length of the audio files, distribution of spoofed files).
    
* show_gates.py plots some some statistics about the weights ofr a saved population 
with neat.Checkpointer. It is only working with raw audio files and neat-python.
We may implement the others a next time

* metrics_utils.py file implements the error equal rate 
    * Implemented by Andreas Nautsch (EURECOM) and Themos Stafylakis (Omilia),
      source code available [here](https://gitlab.eurecom.fr/nautsch/pybosaris)
    * libmath.py contains auxiliary code used by metrics_utils.py.

* neat.cfg configuration file that defines the parameters of the neat algorithm

## Note
To run the code, you have to add a folder data containing LA and PA, in the folder anti spoofing.
The LA and PA folders can be download [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

The code should be runnning both on Linux and Windows machine.

The ease of classification and the grand champion algorithm are from the following paper:

[Valenti, Giacomo and  Delgado, Héctor and  Todisco, Massimiliano and  Evans, Nicholas and  Pilati, Laurent, 
(2018),
*An end-to-end spoofing countermeasure for automatic speaker verification using evolving recurrent neural networks*
](http://www.eurecom.fr/fr/publication/5523/detail/an-end-to-end-spoofing-countermeasure-for-automatic-speaker-verification-using-evolving-recurrent-neural-networks)


## Future Work
