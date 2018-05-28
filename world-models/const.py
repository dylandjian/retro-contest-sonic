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
LATENT_VEC = 128
VAE_LOSS = "bce"

## RNN
HIDDEN_UNITS = 512
HIDDEN_DIM = 1024
TEMPERATURE = 1.25
GAUSSIANS = 8
NUM_LAYERS = 1
MDN_CONST = 1.0 / math.sqrt(2.0 * math.pi)

## Controller
SIGMA_INIT = 0.1
POPULATION = 64
SCORE_CAP = 5000
REPEAT_ROLLOUT = 4
RENDER_TICK = 32
TIMESTEP_DECAY = 150 

## Image size
HEIGHT = 128
WIDTH = 128

## Dataset
SIZE = 17000
REPEAT = 0

## Play
PARALLEL = 4
PLAYOUTS = 2500
PLAYOUTS_PER_LEVEL = 10000
MAX_REPLACEMENT = 1
ACTION_SPACE = 9

## Training
MOMENTUM = 0.9 ## SGD
ADAM = True
LR = 1e-3
L2_REG = 1e-4
LR_DECAY = 0.1
BATCH_SIZE = 32

## Refresh
LOSS_TICK = 100
REFRESH_TICK = 500
SAVE_TICK = 5000
LR_DECAY_TICK = 100000

## Jerk
EXPLOIT_BIAS = 0.25
MAX_TIMESTEPS = 900

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
