import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import time

DEVICE = torch.device("cuda")

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x

class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
   

class Test(mp.Process):
    def __init__(self, model1, model2, controller, idx):
        super(Test, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.controller = controller
        self.idx = idx
    
    def run(self):
        print("Starting: %d" % self.idx)
        while True:
            time.sleep(20)


def train():
    model1 = Model1().to(DEVICE)
    model2 = Model2().to(DEVICE)

    jobs = []
    for idx in range(5):
        controller = Controller().to(DEVICE)
        new_w = torch.randn(1024, dtype=torch.float, device=DEVICE)
        controller.state_dict()['fc1.weight'].data.copy_(new_w)
        new_process = Test(model1, model2, controller, idx)
        jobs.append(new_process)
        new_process.start()
    for p in jobs:
        p.join()

        
if __name__ == "__main__":
    mp.set_start_method('spawn')
    train()
