# NEAT and IRIS

This part is to get familiarized with NEAT and assess its performance in 
a classification task.

Here is the only file that you can execute:
[iris_neat.py](https://github.com/Maxwell1447/Neuro-evolution-in-speaker-recognition/tree/master/iris/iris_neat.py)

You can also check the images and graphs of some results in 
[this folder](https://github.com/Maxwell1447/Neuro-evolution-in-speaker-recognition/tree/master/iris/result%20varieties).


In the main part of the script (at the very bottom), you can change various parameters:

* **Features**: To change the features taken into account, choose a subset of ```[0, 1, 2, 3]```. 
They correspond to petal/sepal length/width.
Note that if you want to plot the decision surface, you need to choose a subset of size 2.
```python
features = [0, 2]         
```

* **Wrong labelling**: You can choose to label randomly a random subset of data points with the ```wrong_labelling``` parameter.
It is ideal to test the robustness of NEAT to errors and noise in terms of classification.
```python
data, labels, names = load_iris(features, wrong_labelling=15)
```

* **Run NEAT once**: Uncomment these lines to run NEAT once. You should be able to visualize the fitness evolution,
the winner's topology, and the surface plot (provided you chose 2 features exactly). You can control the 
max number of generation that is here set to 50.
```python
random.seed(0)
winner, config, stats, acc = run(config_path, 50)
make_visualize(winner, config, stats, decision_surface=(len(features) == 2))
```

* **Run NEAT sevral times**: Uncomment this line to run NEAT several times (25 times here), and visualize the 
histogram of the number of generations required to reach the threshold, plus the histogram of the testing accuracy.

```python
general_stats(25, config_path)
```
---

## Results

### Simple run + no wrong labelling
When plotting the decision surface, we can see that NEAT correctly classifies the IRIS data points.

![](./result%20varieties/decision_surface.svg)

The kind of topology obtained can be very simple:

![](./result%20varieties/topology.JPG)

### Simple run + 10 wrong labelling
We can observe that NEAT still manages to correctly separate the clusters. 

![](./result%20varieties/decision_surface%20noisy%202%20features.svg)

### General stats + no wrong labelling

Number of generation before reaching fitness threshold:

![](./result%20varieties/generation%20histogram%204%20features.svg)

![](./result%20varieties/accuracy%20repartition%204%20features.png)

# Details about the code

The most important part of the code is probably the function ```eval_genomes()``` in which we calculate 
the fitness of each genome. The fitness is chosen to be *30 - MSE* so that the threshold of 0 can be reached.

We chose to use the tensorial evaluation provided by PytorchNEAT for the sake of speed. 

The [config file](https://github.com/Maxwell1447/Neuro-evolution-in-speaker-recognition/tree/master/iris/config_iris)
can be modified manually to see the influence of the parameters. We empirically experienced that 
*node_add_prob* was one of the most important parameters, along with *conn_add_prob*, *weight_mutate_power* and *bias_mutate_power*.
More global statistics should be done (with Optuna for instance) to assess the contribution of all the parameters.s

