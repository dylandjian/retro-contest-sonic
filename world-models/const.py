import torch
import math

##### CONFIG

## CUDA variable from Torch
CUDA = torch.cuda.is_available()
## Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")

## Eps for log
EPSILON = 1e-6

## VAE
LATENT_VEC = 64
VAE_LOSS = "bce"

## RNN
HIDDEN_UNITS = 256
HIDDEN_DIM = 1024
GAUSSIANS = 10
NUM_LAYERS = 4

## Image size
HEIGHT = 128
WIDTH = 128

## Dataset
SIZE = 5000
REPEAT = 0

## Play
PARALLEL = 5
PLAYOUTS = 2500
MAX_REPLACEMENT = 0.4

## Training
MOMENTUM = 0.9 ## SGD
ADAM = True
LR = 1e-3
L2_REG = 1e-4
LR_DECAY = 0.1
BATCH_SIZE = 32

## Refresh
LOSS_TICK = 50
REFRESH_TICK = 1000
SAVE_TICK = 1000
LR_DECAY_TICK = 100000

## Jerk
EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = 1000000
MDN_CONST = 1.0 / math.sqrt(2.0 * math.pi)

## Env
GAMES = {
    "SONIC-1": "SonicTheHedgehog-Genesis"
}
LEVELS = {
    "SonicTheHedgehog-Genesis": [
        "SpringYardZone.Act3",
        "SpringYardZone.Act2",
        "GreenHillZone.Act3",
        "GreenHillZone.Act1",
        "StarLightZone.Act2",
        "StarLightZone.Act1",
        "MarbleZone.Act2",
        "MarbleZone.Act1",
        "MarbleZone.Act3",
        "ScrapBrainZone.Act2",
        "LabyrinthZone.Act2",
        "LabyrinthZone.Act1",
        "LabyrinthZone.Act3"
    ]
}
