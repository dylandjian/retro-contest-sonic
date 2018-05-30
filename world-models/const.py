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
LATENT_VEC = 200
BETA = 4
VAE_LOSS = "bce"

## RNN
HIDDEN_UNITS = 1024
HIDDEN_DIM = 1024
TEMPERATURE = 1.25
GAUSSIANS = 8
NUM_LAYERS = 1
MDN_CONST = 1.0 / math.sqrt(2.0 * math.pi)

## Controller
PARALLEL = 5
SIGMA_INIT = 0.1
POPULATION = 70
SCORE_CAP = 8000
REPEAT_ROLLOUT = 6
RENDER_TICK = 10
MAX_TIMESTEPS = 1800
TIMESTEP_DECAY = 150 
TIMESTEP_DECAY_TICK = 5
REWARD_BUFFER = 450
MIN_REWARD = 20

## Image size
HEIGHT = 128
WIDTH = 128

## Dataset
SIZE = 17000
REPEAT = 0

## Play
PARALLEL_PER_GAME = 2
PLAYOUTS = 2500
PLAYOUTS_PER_LEVEL = 10000
MAX_REPLACEMENT = 0.2
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
REFRESH_TICK = 1000
SAVE_PIC_TICK = 500
SAVE_TICK = 5000
LR_DECAY_TICK = 100000

## Jerk
EXPLOIT_BIAS = 0.25
TOTAL_TIMESTEPS = 1e6

## Env
GAMES = [
    "SonicTheHedgehog-Genesis",
    "SonicTheHedgehog2-Genesis",
    "SonicAndKnuckles3-Genesis"
]

LEVELS = {
    "SonicTheHedgehog-Genesis": [
        "GreenHillZone.Act1",
        "GreenHillZone.Act2",
        "GreenHillZone.Act3",
        "SpringYardZone.Act1",
        "SpringYardZone.Act2",
        "SpringYardZone.Act3",
        "StarLightZone.Act1",
        "StarLightZone.Act2",
        "StarLightZone.Act3",
        "MarbleZone.Act1",
        "MarbleZone.Act2",
        "MarbleZone.Act3",
        "ScrapBrainZone.Act1",
        "ScrapBrainZone.Act2",
        "LabyrinthZone.Act1",
        "LabyrinthZone.Act2",
        "LabyrinthZone.Act3"
    ],
    "SonicTheHedgehog2-Genesis": [
        "EmeraldHillZone.Act1",
        "EmeraldHillZone.Act2",
        "ChemicalPlantZone.Act1",
        "ChemicalPlantZone.Act2",
        "MetropolisZone.Act1",
        "MetropolisZone.Act2",
        "MetropolisZone.Act3",
        "OilOceanZone.Act1",
        "OilOceanZone.Act2",
        "MysticCaveZone.Act1",
        "MysticCaveZone.Act2",
        "HillTopZone.Act1",
        "HillTopZone.Act2",
        "CasinoNightZone.Act1",
        "CasinoNightZone.Act2",
        "AquaticRuinZone.Act2",
        "AquaticRuinZone.Act1",
        "WingFortressZone"
    ],
    "SonicAndKnuckles3-Genesis": [
        "LavaReefZone.Act1",
        "LavaReefZone.Act2",
        "CarnivalNightZone.Act1",
        "CarnivalNightZone.Act2",
        "MarbleGardenZone.Act1",
        "MarbleGardenZone.Act2",
        "MushroomHillZone.Act1",
        "MushroomHillZone.Act2",
        "DeathEggZone.Act1",
        "DeathEggZone.Act2",
        "FlyingBatteryZone.Act1",
        "FlyingBatteryZone.Act2",
        "SandopolisZone.Act1",
        "SandopolisZone.Act2",
        "HydrocityZone.Act1",
        "HydrocityZone.Act2",
        "IcecapZone.Act1",
        "IcecapZone.Act2",
        "AngelIslandZone.Act1",
        "AngelIslandZone.Act2",
        "LaunchBaseZone.Act1",
        "LaunchBaseZone.Act2",
        "HiddenPalaceZone"
    ]
}

LEVELS_VALID = {
    "SonicTheHedgehog-Genesis": [
        "SpringYardZone.Act1",
        "GreenHillZone.Act2",
        "StarLightZone.Act3",
        "ScrapBrainZone.Act1"
    ],
    "SonicTheHedgehog2-Genesis": [
        "MetropolisZone.Act3",
        "HillTopZone.Act2",
        "CasinoNightZone.Act2"
    ],
    "SonicAndKnuckles3-Genesis": [
        "LavaReefZone.Act1",
        "FlyingBatteryZone.Act2",
        "HydrocityZone.Act1",
        "AngelIslandZone.Act2"
    ]
}



