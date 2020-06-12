# Anti spoofing
This is the anti spoofing folder. Here are the scripts for running neat on the ASVspoof 2019 databse.

## Table of contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Features](#features)

## General info
We are using the ASVspoof 2019 logical (LA) database.
The logical train audio files are used for training.
The logical dev audio files are used for testing.

## Screenshots
![Example screenshot](./img/Digraph.gv.svg)

A topology of a genome obtained by running neat on the ASVspoof 2019 database.

## Technologies
* neat 0.92



## Code Examples
Show examples of usage:
```	
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

## Features
List of features ready and TODOs for future development
* Awesome feature 1
* Awesome feature 2
* Awesome feature 3

To-do list:
* Wow improvement to be done 1
* Wow improvement to be done 2
