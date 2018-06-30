# retro-contest-sonic

A student implementation of the World Models paper with documentation.

Ongoing project.


# TODO


# CURRENTLY DOING


# DONE

* β-VAE for the Visual model
* MDN-LSTM for the Memory model
* CMA-ES for the Controller model
* Training pipelines for the 3 models
* Human recordings to generate data
* MongoDB to store data
* LSTM and VAE trained "successfully"
* Multiprocessing of the evaluation of a set of parameters given by the CMA-ES
* Submit learnt agents


# LONG TERM PLAN ?

* Cleaner code, more optimized and documented
* Game agnostic
* Continue training / testing better architectures
* Online training instead of using a database

# How to launch the scripts

- Install the modules in the requirements.txt.    
- Buy/Find the ROMs of Sonic The Hedgehog and install them with retro-gym.

The github repo contains a trained VAE & LSTM and a *really* bad controller that has been trained for 30 minutes.
Once you've done that, you can either retrain the controller (or the VAE / LSTM, just replace train_controller by train_xxx in the following command) from scratch by deleting the controller and the solver saved parameters in the folder saved_models/1527608256, or play a random level with the current saved models using this command   
   
`python play_best.py --folder=1527608256`   
or if you want the controller model :
`python train_controller.py --folder=1527608256`


# Resources

* [My write-up on the code and concepts of this repository](https://dylandjian.github.io/world-models/)
* [World Models paper](https://arxiv.org/pdf/1803.10122.pdf)
* Coded on Ubuntu 16.04, Python 3.5, PyTorch 0.4 with GPUs (some change have to be made in order to make it fully compatible to CPU as well, such as adding a map_location="cpu" when loading the model)


# Differences with the official paper

* No temperature
* No flipping of the loss sign during training (to encourage exploration)
* β-VAE instead of VAE
