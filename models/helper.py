import os
from models.vae import VAE, ConvVAE
from models.lstm import LSTM
from models.controller import Controller
import torch
from const import *
import pickle


def save_checkpoint(model, filename, state, current_time):
    """ Save a checkpoint of the models """

    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                        '..', 'saved_models', current_time)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = os.path.join(dir_path, "{}-{}.pth.tar"\
                        .format(state['version'], filename))
    state['model'] = model.state_dict()
    torch.save(state, filename)


def load_torch_models(path, model, filename):
    """ Load an already saved model """

    checkpoint = torch.load(os.path.join(path, filename))
    model.load_state_dict(checkpoint['model'])
    return checkpoint


def get_version(folder_path, file_version, model):
    """ Either get the last versionration of 
        the specific folder or verify it version exists """

    if int(file_version) == -1:
        files = os.listdir(folder_path)
        files = list(filter(lambda x: model in x, files))
        if len(files) > 0:
            all_version = list(map(lambda x: int(x.split('-')[0]), files))
            all_version.sort()
            file_version = all_version[-1]
        else:
            return False
    if model != "solver":
        test_file = "{}-{}.pth.tar".format(file_version, model)
    else:
        test_file = "{}-{}.pkl".format(file_version, model)
    if not os.path.isfile(os.path.join(folder_path, test_file)):
        return False
    return file_version


def load_model(folder, version, model="vae"):
    """ Load a player given a folder and a version """

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                   '..', 'saved_models')
    if folder == -1:
        folders = os.listdir(path)
        folders.sort()
        if len(folders) > 0:
            folder = folders[-1]
        else:
            return False, False
    elif not os.path.isdir(os.path.join(path, str(folder))):
        return False, False

    folder_path = os.path.join(path, str(folder))
    last_version = get_version(folder_path, version, model)
    if model == "controller":
        solver_version = get_version(folder_path, version, "solver")
    else:
        solver_version = None
    if not last_version:
        return False, False

    return get_player(folder, int(last_version), model, solver_version=solver_version)


def get_player(current_time, version, file_model, solver_version=None):
    """ Load the models of a specific player """

    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                            '..', 'saved_models', str(current_time))
    try:
        mod = os.listdir(path)
        models = list(filter(lambda model: (model.split('-')[0] == str(version) \
                        and file_model in model), mod))
        models.sort()
        if len(models) == 0:
            return False, version
    except FileNotFoundError:
        return False, version
    
    if file_model == "vae": 
        model = ConvVAE((HEIGHT, WIDTH, 3), LATENT_VEC).to(DEVICE)
    elif file_model == "lstm":
        model = LSTM(HIDDEN_UNITS, LATENT_VEC,\
                     NUM_LAYERS, GAUSSIANS, HIDDEN_DIM).to(DEVICE)
    elif file_model == "controller":
        model = Controller(LATENT_VEC, HIDDEN_UNITS * NUM_LAYERS * 2,
                            ACTION_SPACE).to(DEVICE)

    checkpoint = load_torch_models(path, model, models[0])
    if file_model == "controller":
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                    '..', 'saved_models', current_time, "{}-solver.pkl".format(solver_version))
        solver = pickle.load(open(file_path, 'rb'))
        return checkpoint, model, solver
    return model, checkpoint
