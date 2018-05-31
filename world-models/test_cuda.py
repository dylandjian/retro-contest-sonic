import torch.multiprocessing as multiprocessing
from models.controller import Controller
from const import *
import time



def test(controller):
    print("cc")
    new_ctrl = controller
    while True:
        time.sleep(20)




p = []
for i in range(2):
    best_controller = Controller(LATENT_VEC, HIDDEN_UNITS * NUM_LAYERS * 2,
                                        ACTION_SPACE).to(DEVICE)
    j = multiprocessing.Process(target=test, args=(best_controller,))
    j.start()
    p.append(j)
for j in p:
    j.join()