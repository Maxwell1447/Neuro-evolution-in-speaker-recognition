# Raw audio gender classification with NEAT

The results obtained so far have been rather poor. This part may be updated were we to obtain good results
on anti-spoofing. In this case, we could change the parts that did not work.

## Previous work on raw audio gender classification

The project from which we fetched the database and the scripts for pre-processing used another method of 
machine learning: 1D-CNNs. In our case we have a clearly different approach with NEAT: we have recurrent network topologies.
This RNN inspired method can be quite unstable due to vanishing/exploding gradient issues.
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

You can run [this script (main.py)](main.py) to see the results (change the number of generation in main to decrease the 
execution time is it is too long). You can also change different parameters in the 
[config file](neat.cfg) if necessary.

##### Remarks about the code

The evaluation is parallelized by default with ```GenderEvaluator```. If you want to see the difference of execution time
by yourself, change the following line in ```run()``` by the next one:

```python
winner_ = p.run(multi_evaluator.evaluate, n_gen)
```
```python
winner_ = p.run(eval_genomes, n_gen)
```

You can also try to uncomment the quote comment in ```run()``` to see if you manage to run 
NEAT on partial data only. There were some referencing issues due to parallelization and global values,
so it should most probably not work.