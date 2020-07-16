# Raw audio gender classification with NEAT

The results obtained so far have been rather poor with raw audio. Nonetheless, the simple usage of preprocessing techniques
like STFT, CQT, MFCC were sufficient to obtain way better results.

## Previous work on raw audio gender classification

The project from which we fetched the database and the scripts for pre-processing used another method of 
machine learning: 1D-CNNs. In our case we have a clearly different approach with NEAT: we have recurrent network topologies.
This RNN inspired method can be quite unstable on raw audio due to vanishing/exploding gradient issues.
We nonetheless tried to implement NEAT, being inpired from [this paper](http://www.eurecom.fr/fr/publication/5523/download/sec-publi-5523_3.pdf).

## RNN

Before using NEAT, we attempted to test RNN networks that correspond to recurrent fixed topologies of NEAT phenotypes.

You can run [this script (rnn.py)](rnn.py) to train either RNN, GRU, LSTM or even ConvNet nets (the last one being the one from the original project).

To change the network that is trained, look at the main of the script, and change the following line:
```python
model = "GRU"  # or "RNN", "LSTM", "ConvNet"
```

You will most likely obtain no performance increase during the training, except for ```ConvNet```.


## NEAT

You can run [this script (main.py)](main.py) to see the results with no preprocessing (change the number of generation in main to decrease the 
execution time is it is too long). You can also change different parameters in the 
[config file](neat.cfg) if necessary.

You will most likely obtain poor results with fitness stagnation, and an accuracy not better than randomness.

With preprocessing, you can run [this script (neat_preprocessed.py)](neat_preprocessed.py), associated with
[this config file](neat_preprocessed.cfg). This will generate and save data loader (train + test) of around 
0.25GB.
Note also that there is a testing accuracy reporter that compute the testing accuracy of the best genome every
100 generations.

#### Visualization

If you wish to see the behavior of the gate of the best genome obtained by [(neat_preprocessed.py)](neat_preprocessed.py)]
you can run [(visualize_behavior.py)](visualize_behavior.py). You can then see the signal w.r.t. the gate, and 
assess its relevance (does it consider silent parts).

#### Remarks about the code

The evaluation is parallelized by default with ```GenderEvaluator```. If you want to see the difference of execution time
by yourself, change the following line in ```run()``` by the next one:

```python
winner_ = p.run(multi_evaluator.evaluate, n_gen)
```
```python
winner_ = p.run(eval_genomes, n_gen)
```

