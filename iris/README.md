# NEAT and IRIS

This part is to get familiarized with NEAT and assess its performance in 
a classification task.

Here is the only file that you can execute:
[iris_neat.py](https://github.com/Maxwell1447/Neuro-evolution-in-speaker-recognition/tree/master/iris/iris_neat.py)

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
the winner's topology, and the surface plot (provided you chose 2 features exactly).
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

<img src="https://github.com/Maxwell1447/Neuro-evolution-in-speaker-recognition/tree/master/iris/result varieties/decision_surface 2.svg"  alt="image"/> 

