# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import multiprocessing
import os

import click
import neat
import neat_local.visualization.visualize as visualize

# import torch
import numpy as np

from neat_local.pytorch_neat import t_maze
from neat_local.pytorch_neat.activations import tanh_activation
from neat_local.pytorch_neat.adaptive_linear_net import AdaptiveLinearNet
from neat_local.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from neat_local.pytorch_neat.neat_reporter import LogReporter

batch_size = 4
DEBUG = False

os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\graphviz\\bin"


def make_net(genome, config, _batch_size):
    input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    output_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    return AdaptiveLinearNet.create(
        genome,
        config,
        input_coords=input_coords,
        output_coords=output_coords,
        weight_threshold=0.4,
        batch_size=batch_size,
        activation=tanh_activation,
        output_activation=tanh_activation,
        device="cpu",
    )


def activate_net(net, states, debug=False, step_num=0):
    if debug and step_num == 1:
        print("\n" + "=" * 20 + " DEBUG " + "=" * 20)
        print(net.delta_w_node)
        print("W init: ", net.input_to_output[0])
    outputs = net.activate(states).numpy()
    if debug and (step_num - 1) % 100 == 0:
        print("\nStep {}".format(step_num - 1))
        print("Outputs: ", outputs[0])
        print("Delta W: ", net.delta_w[0])
        print("W: ", net.input_to_output[0])
    return np.argmax(outputs, axis=1)


'''
@click.command()
@click.option("--n_generations", type=int, default=20)
@click.option("--n_processes", type=int, default=6)
'''
def run(n_generations, n_processes):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.dirname(__file__)+"/neat.cfg"
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    envs = [t_maze.TMazeEnv(init_reward_side=i, n_trials=100) for i in [1, 0, 1, 0]]

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, envs=envs, batch_size=batch_size, max_env_steps=1000
    )

    if n_processes > 1:
        pool = multiprocessing.Pool(processes=n_processes)

        def eval_genomes(genomes, config):
            fitnesses = pool.starmap(
                evaluator.eval_genome, ((genome, config) for _, genome in genomes)
            )
            for (_, genome), fitness in zip(genomes, fitnesses):
                genome.fitness = fitness

    else:

        def eval_genomes(genomes, config):
            for i, (_, genome) in enumerate(genomes):
                try:
                    genome.fitness = evaluator.eval_genome(
                        genome, config, debug=DEBUG and i % 100 == 0
                    )
                except Exception as e:
                    print(genome)
                    raise e

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("log.json", evaluator.eval_genome)
    pop.add_reporter(logger)

    winner = pop.run(eval_genomes, n_generations)

    print(winner)
    final_performance = evaluator.eval_genome(winner, config)
    print("Final performance: {}".format(final_performance))
    generations = reporter.generation + 1
    
    node_names = {-1:'left', -2: 'front', -3:'right', -4:'color'}
    node_names[-5] = "reward"
    node_names[-6] = "False"
    node_names[-7] = "{}"
    node_names[0] = "direction"
    
    # Visualization
    visualize.draw_net(config, winner, True, filename="graph_neat_examples_T-maze", node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True, filename="stats_neat_examples_T-maze")
    visualize.plot_species(stats, view=True, filename="species_neat_examples_T-maze")
    
    input_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, -1.0]]
    output_coords = [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
    
    return AdaptiveLinearNet.create(
        winner,
        config,
        input_coords=input_coords,
        output_coords=output_coords,
        weight_threshold=0.4,
        batch_size=1,
        activation=tanh_activation,
        output_activation=tanh_activation,
        device="cpu",
    )

    

if __name__ == "__main__":
    winner_net = run(5, 1)  # pylint: disable=no-value-for-parameter
    print("\n \n")
    env = t_maze.TMazeEnv()
    nb_episode = 5
    sum_reward = 0
    for i_episode in range(nb_episode):
        states = [env.reset()]
        #inputs = env.state(), 0, False, {}
        for t in range(100):
            print(t)
            env.render()
            [action] = activate_net(winner_net, states, debug=DEBUG, step_num=t)
            states = [np.array(env.step(action)[0])]
    env.close()
    
