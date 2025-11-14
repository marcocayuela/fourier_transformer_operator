import os
from training.dataset_manager import DatasetManager
from training.trainer import Trainer
from factory import Factory
from models import FTO
import torch

class FTOTraining():

    def __init__(self, args):

        #Device parameters
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps") 
        else:
            self.device = torch.device("cpu")
        
        self.args = args

        # File and repo managning 
        self.exp_dir = self.args["exp_dir"]
        self.exp_name = self.args["exp_name"]

        # Training parameters
        self.seq_length = self.args["seq_length"]
        self.batch_size = self.args["batch_size"]
        self.num_workers = self.args["num_workers"]
        self.loss_fn = self.args["loss_fn"]
        self.optimizer_info = self.args["optimizer"]
        self.num_epochs = self.args["num_epochs"]
        self.scheduler_info = self.args["scheduler"]
        self.metrics_name = self.args["metrics"]

        # Datasets
        self.datasets = DatasetManager(self.exp_dir, self.seq_length, self.batch_size, self.num_workers)

        # Model definition
        self.name_weights_to_load = self.args["name_weights_to_load"]

        self.input_dim = self.args["input_dim"]
        self.output_dim = self.args["output_dim"]
        self.n_dim = self.args["n_dim"]
        self.domain_size = self.args["domain_size"]
        self.modes_separation = self.args["modes_separation"]
        self.norm_separation = self.args["norm_separation"]

        self.representation_dim = self.args["representation_dim"]

        self.n_heads = self.args["n_heads"]
        self.n_attblocks = self.args["n_attblocks"]
        self.hidden_dim = self.args["hidden_dim"]


    def make_directories(self):

        directories = [os.path.join('runs',self.exp_dir),
                       os.path.join('runs',self.exp_dir, self.exp_name),
                       os.path.join('runs',self.exp_dir, self.exp_name, 'model_weights'),
                       os.path.join('runs',self.exp_dir, self.exp_name, 'logs')]
        
        for d in directories:
            os.makedirs(d, exist_ok=True)


    def execute_experience(self):


        # Dataloaders creation
        self.datasets = DatasetManager(self.exp, self.seq_length, self.batch_size, self.num_workers)

        # Directories creation
        self.make_directories()

        # Load or create model
        model = FTO(self.input_dim, self.output_dim, self.representation_dim,
                    self.device, self.modes_separation, self.n_dim,
                    self.domain_size, self.norm, self.seq_len, self.n_heads,
                    self.n_attblocks, self.hidden_dim)

        if self.name_weights_to_load is not None:
            path_model = os.path.join('runs',self.exp_dir,self.exp_name,'model_weights')
            loaded_weights = torch.load(os.path.join(path_model, self.name_weights_to_load))
            self.last_epoch = loaded_weights['epoch']
            state_dict = loaded_weights['model_state_dict']
            model.load_state_dict(state_dict)

        self.optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(), lr=self.optimizer_info["lr"])
        self.scheduler = Factory.get_scheduler(self.scheduler_info, self.optimizer, max_lr=self.optimizer_info["lr"], n_epoch=self.num_epochs, n_batch=self.datasets.batch_size)
        self.metrics = [Factory.get_metric(metric) for metric in self.metrics_name]
        # Create the trainer and train

        trainer = Trainer(model=model,
                          train_loader=self.datasets.training_loader,
                          test_loader=self.datasets.testing_loader,
                          loss_fn=self.loss_fn,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          num_epochs=self.num_epochs,
                          device=self.device,
                          exp_dir=self.exp_dir,
                          exp_name=self.exp_name, 
                          metrics=self.metrics)
        
        trainer.train_loop()


