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

import os

import click
import gym
import neat
import neat_local.visualization.visualize as visualize

from neat_local.pytorch_neat.multi_env_eval import MultiEnvEvaluator
from neat_local.pytorch_neat.neat_reporter import LogReporter
from neat_local.pytorch_neat.recurrent_net import RecurrentNet

from neat.nn.feed_forward import FeedForwardNetwork


max_env_steps = 10000



def make_env():
    env = gym.make("CartPole-v1")
    env._max_episode_steps = max_env_steps
    return env


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs[:, 0] > 0.5

'''
@click.command()
@click.option("--n_generations", type=int, default=5)
'''
def run(n_generations):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    evaluator = MultiEnvEvaluator(
        make_net, activate_net, make_env=make_env, max_env_steps=max_env_steps
    )

    def eval_genomes(genomes, config):
        for _, genome in genomes:
            genome.fitness = evaluator.eval_genome(genome, config)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    logger = LogReporter("neat_local.log", evaluator.eval_genome)
    pop.add_reporter(logger)

    winner = pop.run(eval_genomes, n_generations)
    
    node_names = {-1:'Cart Position', -2: 'Cart Velocity', -3:'Pole Angle', -4:'Pole Velocity At Tip', 0:'Push cart to the left or to the right'}
    
    
    # Visualization
    visualize.draw_net(config, winner, True, filename="graph_neat_examples_CartPole-v1", node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True, filename="stats_neat_examples_CartPole-v1")
    visualize.plot_species(stats, view=True, filename="species_neat_examples_CartPole-v1")
    
    

    winner_net = FeedForwardNetwork.create(winner, config)
    
    return winner_net 



if __name__ == "__main__":
    winner_net = run(15)  # pylint: disable=no-value-for-parameter
    print("\n \n")
    env = make_env()
    nb_episode = 5
    sum_timesteps = 0
    for i_episode in range(nb_episode):
        observation = env.reset()
        action = env.action_space.sample()
        for t in range(max_env_steps):
            env.render()
            observation, reward, done, info = env.step(action)
            action = winner_net.activate(observation)[0]
            if action > .5:
                action = 1
            else:
                action = 0
            if done:
                print("\n Episode finished after {} timesteps".format(t+1))
                sum_timesteps += t+1
                break
    env.close()
    print("\n average timesteps = ", sum_timesteps/ nb_episode )
    