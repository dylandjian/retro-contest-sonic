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

- Install the modules in the requirements.txt, pytorch 0.4 and mongoDB   
- Buy or find the ROMs of Sonic The Hedgehog and install them with retro-gym.

Once you've done that, you will need to train the 3 components :   
`python train_vae.py`   
`python train_lstm.py --folder=xxx`   
`python train_controller.py --folder=xxx`  where xxx is the folder number created in saved_models/   
   
While training the VAE and the LSTM, pictures will be saved in a folder results/   
    
Once you're done, you can use your best trained controller to play a random level using :
`python play_best --folder=xxx`   
Dont forget to change the RENDER_TICK in const.py to 1, so you can see what's happening.

# Resources

* [My write-up on the code and concepts of this repository](https://dylandjian.github.io/world-models/)
* [World Models paper](https://arxiv.org/pdf/1803.10122.pdf)
* Coded on Ubuntu 16.04, Python 3.5, PyTorch 0.4 with GPUs (some change have to be made in order to make it fully compatible to CPU as well, such as adding a map_location="cpu" when loading the model)


# Differences with the official paper

* No temperature
* No flipping of the loss sign during training (to encourage exploration)
* β-VAE instead of VAE
