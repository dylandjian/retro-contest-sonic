import torch

##### CONFIG

## CUDA variable from Torch
CUDA = torch.cuda.is_available()
## Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")
## Latent vector in VAE size
LATENT_VEC = 40

HEIGHT = 224
WIDTH = 320
SIZE = 2000
REPEAT = 4
PARALLEL = 3
PLAYOUTS = 1000

LR = 1e-3

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
