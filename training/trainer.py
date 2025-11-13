import torch

import os 
import time

class Trainer():

    def __init__(self, model, optimizer, loss_fn, scheduler, device="cpu"):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device

    
    