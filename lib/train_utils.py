from const import *
import pickle
import torch
from models.vae import ConvVAE
from models.lstm import LSTM
from models.controller import Controller
from .controller_utils import CMAES
from models.helper import load_model, save_checkpoint


def init_models(current_time, load_vae=True, load_lstm=True, load_controller=True):
    vae = lstm = best_controller = solver = None

    if load_vae:
        vae, _ = load_model(current_time, -1, model="vae")
        if not vae:
            vae = ConvVAE((HEIGHT, WIDTH, 3), LATENT_VEC).to(DEVICE)
    
    if load_lstm:
        lstm, _ = load_model(current_time, -1, model="lstm")
        if not lstm:
            lstm = LSTM(HIDDEN_UNITS, LATENT_VEC,\
                        NUM_LAYERS, GAUSSIANS, HIDDEN_DIM).to(DEVICE)
    
    if load_controller:
        res = load_model(current_time, -1, model="controller")
        checkpoint = res[0]
        if len(res) > 2:
            best_controller = res[1]
            solver = res[2]
            current_ctrl_version = checkpoint['version']
        else:
            best_controller = Controller(LATENT_VEC, PARAMS_FC1,
                                            ACTION_SPACE).to(DEVICE)
            solver = CMAES(PARAMS_FC1 + LATENT_VEC + 512,
                        sigma_init=SIGMA_INIT,
                        popsize=POPULATION)
    
    return vae, lstm, best_controller, solver, checkpoint

def update_lr(lr, optimizer, total_ite, lr_decay=LR_DECAY, lr_decay_tick=LR_DECAY_TICK):
    """ Decay learning rate by a factor of lr_decay every lr_decay_ite iteration """

    if total_ite % lr_decay_tick != 0 or lr <= 0.0001:
        return lr, optimizer
    
    print("[TRAIN] Decaying the learning rate !")
    lr = lr * lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr, optimizer


def fetch_new_run(collection, fs, dataset, last_id, loaded_version=None):
    """ Update the dataset with new run from the databse """

    ## Fetch new run in reverse order so we add the newest run first
    new_run = collection.find({"id": {"$gt": last_id}}).sort("id", 1)
    added_frames = 0
    added_runs = 0
    print("[TRAIN] Fetching: %d new run from the db"% (new_run.count()))

    for run in new_run:
        data = pickle.loads(fs.get(run['run']).read())
        number_frames = dataset.update(data)
        added_frames += number_frames
        added_runs += 1

        ## You cant replace more than 40% of the dataset at a time
        if loaded_version and added_frames > SIZE or \
            added_frames >= SIZE * MAX_REPLACEMENT and not loaded_version:
            break
        
    print("[TRAIN] Last id: %d, added runs: %d added frames: %d"\
                    % (last_id, added_runs, added_frames))
    return last_id + added_runs


def create_optimizer(model, lr, param=None):
    """ Create or load a saved optimizer """

    if ADAM:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_REG)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, \
                        weight_decay=L2_REG, momentum=MOMENTUM)
    
    if param:
        opt.load_state_dict(param)
    
    return opt


def create_state(current_version, lr, total_ite, optimizer):
    """ Create a checkpoint to be saved """

    state = {
        'version': current_version,
        'lr': lr,
        'total_ite': total_ite + 1,
        'optimizer': optimizer.state_dict()
    }
    return state