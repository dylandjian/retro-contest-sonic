import torch
import math


##### CONFIG

torch.set_printoptions(precision=10)
## CUDA variable from Torch
CUDA = torch.cuda.is_available()
#torch.backends.cudnn.deterministic = True
## Dtype of the tensors depending on CUDA
DEVICE = torch.device("cuda") if CUDA else torch.device("cpu")

## Eps for log
EPSILON = 1e-6

## VAE
LATENT_VEC = 200
BETA = 3
VAE_LOSS = "bce"

## RNN
OFFSET = 1
HIDDEN_UNITS = 1024
HIDDEN_DIM = 1024
TEMPERATURE = 1.25
GAUSSIANS = 8
NUM_LAYERS = 1
SEQUENCE = 100
PARAMS_FC1 = HIDDEN_UNITS * NUM_LAYERS * 2
MDN_CONST = 1.0 / math.sqrt(2.0 * math.pi)

## Controller
PARALLEL = 2
SIGMA_INIT = 4
POPULATION = 3
SCORE_CAP = 8000
REPEAT_ROLLOUT = 3
RENDER_TICK = 64
REWARD_BUFFER = 10000
SAVE_SOLVER_TICK = 1
MIN_REWARD = 10

## Image size
HEIGHT = 128
WIDTH = 128

## Dataset
SIZE = 10000
MAX_REPLACEMENT = 0.1
REPEAT = 0

## Play
PARALLEL_PER_GAME = 2
PLAYOUTS = 2500
PLAYOUTS_PER_LEVEL = 10000
ACTION_SPACE = 4
ACTION_SPACE_DISCRETE = 10

## Training
MOMENTUM = 0.9 ## SGD
ADAM = True
LR = 1e-3
L2_REG = 1e-4
LR_DECAY = 0.1
BATCH_SIZE_VAE = 300
BATCH_SIZE_LSTM = 2
SAMPLE_SIZE = 200

## Refresh
LOSS_TICK = 5
REFRESH_TICK = 75
SAVE_PIC_TICK = 20
SAVE_TICK = 500
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



