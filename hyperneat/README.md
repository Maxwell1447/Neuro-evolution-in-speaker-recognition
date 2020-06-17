# HyperNEAT

HyperNEAT is a variation of NEAT that uses CPPNs to indirectly
encode the weights of ANNs. To understand HyperNEAT, check
the [user page](http://eplex.cs.ucf.edu/hyperNEATpage/) associated. 
You can also see [this article](https://towardsdatascience.com/hyperneat-powerful-indirect-neural-network-evolution-fba5c7c43b7b)
that vulgarized it so you can have the intuition of this concept.

## PytorchNEAT

PytorchNEAT provides an implementation of CNN, but only two implementations of phenotypes
that are not even useful in our audio classification task. That is why we must implement
the phenotypes we want by ourselves. 

## CNN-HyperNEAT

As CNNs have proven to be efficient in audio classification tasks, we though about 
using HyperNEAT to encode CNNs. 

* [This paper](https://dl.acm.org/doi/pdf/10.1145/3205455.3205459) explains how you can encode
CNNs with HyperNEAT, and develops the concept of MSS-CNN-HyperNEAT. 

* [This paper](https://arxiv.org/pdf/1312.5355.pdf) compares CNN-HyperNEAT to ANN-HyperNEAT
on MNIST. 

The implementation we used in none of those in the papers above for several reasons:

* The MSS extension leads to considerably greater execution times in the reproduction part of NEAT.
This is due to the fact we have a huge number of outputs (one per set of connections between feature maps).
And overall, there is no visible performance increase experienced so far.

* The implementation using CNN-HyperNEAT on MNIST uses it as a feature extractor only and uses SGD to
optimize another ANN in the pipeline. Not only does it increase the execution time, it also gets us away from
our goal to use NEAT as an end-to-end tool. On the other hand, they do not respect the principle of weight sharing to build
the kernels, which lead to a huge amount of weights to encode.

## MNIST

[MNIST_HyperNEAT.py](MNIST_HyperNEAT.py) is an attempt to use CNN-HyperNEAT on MNIST. It uses Optuna
to try to optimize some parameters in the config file. 

The result is stored in a .csv file.

[mnist_analysis.ipynb](mnist_analysis.ipynb) is a notebook to interact with the results, see the importance of each parameter.
We see that overall, our implementation leads to rather poor performances, since a SGD leads to fitness around 0.4, whereas
CNN-HyperNEAT remains around 0.3.

## Future work

* Consider ES-HyperNEAT, that unlike other HyperNEAT implementations, allows
modular network phenotypes.

* Consider the LEO extension that can easily make the connections of a fixed substrate modular.

## Notes

> The [scratch.py](scratch.py) file is an attempt to run CNN-HyperNEAT on audio. It is noted
> as scratch since it is an aborted work.

> There are different config files done, one for each HyperNEAT paradigm done.
> For example, if you wish to test MSS-CNN-HyperNEAT, you must link to the corresponding 
> config file. You should also change the ```eval_genome()``` to match the 
> CPPN and the phenotype.