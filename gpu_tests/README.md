# GPU/CPU comparison

The goal of this section is to evaluate the different phenotype variations that 
still yield the same results. As we are interested in audio, we focused on recurrent 
nets phenotypes.

2 implementations are available in NEAT-Python and PytorchNEAT already, using only CPU.
We added an implementation using GPU to see if it is interesting to switch to GPU.

* **NEAT Vanilla** is the basic implementation of recurrent networks in NEAT-Python.
    * **Creation**: The nodes are ordered so that a node activation only depends on the previous ones.
    * **Evaluation**: A loop goes through this ordered list.
    
   
* **NEAT CPU** is the implementation made in PytorchNEAT using tensors.
    * **Creation**: The nodes are separated into inputs/hidden/outputs, then a matrix is built for each set of
    connections going from one category of node to another (counting itself since it is recursive). 
    Note that the matrix are sparse since many connections do not exist.
    * **Evaluation**: There is just a feed with matrix multiplication.
    
* **NEAT GPU** is a custom implementation we made of **NEAT CPU** using GPU tensors.

## Custom genomes

To compare the creation/evaluation times, we need to deal with genomes of different sizes
to compare when it is preferable to switch from one phenotype to another.

The idea is to have only 2 parameters to define a genome:
* **N** the number of hidden units
* **D** the proportion of dropout (to avoid fully connected nets)

We can then create populations of genomes that we evaluate.
* **P** is the population size

When we feed the nets with inputs, these are sequences.
* **I** is the sequence length of the input.

There is one last parameter that specifies whether we decide to use multiprocessing 
or single processing in the evaluation. Note that you should not use GPU and multiprocessing
together, since it can considerably increase the execution time.

The next step is to choose see the results for each combinations of those values.

## Statistics table

By running [neat_gpu.py](neat_gpu.py), you can create a .csv file with the results for each combination.

The possible values of the 4 parameters are specified in the main:
```python
N = [10, 20, 50, 80, 100]
D = [0.0, 0.5, 0.8]
I = [100, 500, 1000, 2000]
P = [20, 50, 100]
```

The file created is [time_stats_local.csv](time_stats_local.csv). 
To directly visualize the comparison, run [data_analysis.py](data_analysis.py).

The 3D graphics represent the execution time (in z) w.r.t. **I** and **N** of the different 
choices of phenotypes, and whether or not we use multiprocessing (M and S respectively).
> Note that we fixe **P** and **D** here, which that have predictable effects, and are rather constant
> along the NEAT algorithm. Here *P=100* and *D=0.5*.

* The first graphic shows that GPU is never nearly as good as vanilla or CPU.
* The 2 next graphics show that multiprocessing (with 6 cores here) is always better (provided 
we have a large enough input sequence, which is always the case in audio).
* The last graphic shows that for small topologies, vanilla is better, whereas for
bigger ones, CPU overtakes vanilla in terms of rapidity. This suggest that switching from 
vanilla to CPU can be interesting when the topology is big enough.

---

There is [plot_performance.py](plot_performance.py) that show the results of a manual linear
regression of the parameters according to time complexity calculus.

By executing this script, you should be able to visualize the best 
phenotype for a given *(C, I)* couple, where C corresponds to the number of 
connections in the topology.

You should see that GPU is best in a very narrow range of number of connection that 
is too high to be achieved in practice. We also see the result that is may be 
interesting to switch from Vanilla to CPU when reaching a certain 
amount of connections.

### /!\
> The high values of C come from extrapolations of the linear regression, that may not be applicable
> with such extreme values. But the model is relevant in the range of values
> we obtained the results.