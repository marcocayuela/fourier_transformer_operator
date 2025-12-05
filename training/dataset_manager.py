import numpy as np
import os 
import h5py

from training.sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader

class DatasetManager():

    def __init__(self, data_rep, exp_dir, seq_length, batch_size, num_workers, ratio=1):

        self.exp_dir = exp_dir
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_dir = None
        self.training_loader = None
        self.testing_loader = None
        
        if type(ratio)!=int or ratio <1:
            print("Ratio must be an integer greater than 1. It has been automatically set to 1")
            self.ratio = 1
        else:
            self.ratio = ratio  

        if self.exp_dir == "kolmogorov/34":
            data_dir = os.path.join(data_rep,'kolmogorov/34')
            
            train_path = os.path.join(data_dir,'train.h5')
            with h5py.File(train_path, "r") as f:
                self.training_set = f["velocity_field"][()][::self.ratio]
                self.simulation_time_train = f["time"][()][-1] + f["dt"][()]
                self.dt = self.simulation_time_train/self.training_set.shape[0]*self.ratio
                self.n_dim = f["ndim"][()]
                self.nf = f["nf"][()]
                self.nk = f["nk"][()]
                self.re = f["re"][()]
                self.resolution = f["resolution"][()]
                

            test_path = os.path.join(data_dir,'test.h5')
            with h5py.File(test_path, "r") as f:
                self.testing_set = f["velocity_field"][()][::self.ratio]
                self.simulation_time_test = f["time"][()][-1] + f["dt"][()]

    
        self.training_sequence_dataset = SequenceDataset(self.training_set, seq_length=self.seq_length)
        self.testing_sequence_dataset = SequenceDataset(self.testing_set, seq_length=self.seq_length)
        
        self.training_loader = DataLoader(self.training_sequence_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          #num_workers=self.num_workers,
                                          pin_memory=True,
                                          #persistent_workers=True
                                          )

        self.testing_loader = DataLoader(self.testing_sequence_dataset,
                                         batch_size=self.batch_size,
                                         shuffle=False,
                                         #num_workers=self.num_workers,
                                         pin_memory=True,
                                         #persistent_workers=True
                                         )
        
        self.n_batch_train = len(self.training_loader)
        self.n_batch_test = len(self.testing_loader)