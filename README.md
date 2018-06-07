# retro-contest-sonic

A student implementation of the World Models paper with documentation.

Ongoing project.


# TODO


# CURRENTLY DOING

* Submit learnt agents
* Improving the controller training and model to get a decent transfer between levels

# DONE

* β-VAE for the Visual model
* MDN-LSTM for the Memory model
* CMA-ES for the Controller model
* Training pipelines for the 3 models
* Human recordings to generate data
* MongoDB to store data
* LSTM and VAE trained "successfully"
* Multiprocessing of the evaluation of a set of parameters given by the CMA-ES


# LONG TERM PLAN ?

* Cleaner code, more optimized and documented
* Game agnostic
* Online training instead of using a database


# Resources

* [My write-up on the code and concepts of this repository](https://dylandjian.github.io/world-models/)
* [World Models paper](https://arxiv.org/pdf/1803.10122.pdf)
* Coded on Ubuntu 16.04, Python 3.5, PyTorch 0.4 with GPUs (some change have to be made in order to make it fully compatible to CPU as well, such as adding a map_location="cpu" when loading the model)


# Differences with the official paper

* No temperature
* No flipping of the loss sign during training (to encourage exploration)
* β-VAE instead of VAE
