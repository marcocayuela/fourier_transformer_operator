import os
from training.dataset_manager import DatasetManager
from training.trainer import Trainer
from experiments.factory import Factory
from models.FTO import FTO
import torch
import yaml

import time



DATA_DIR = os.getenv("DATA_DIR", "./data")
LOG_DIR = os.getenv("LOG_DIR", "./runs")


class FTOTraining():

    def __init__(self, args):


        self.device_asked = args.get("device", "auto")
        #Device parameters
        if self.device_asked in ["cuda","auto"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.device_asked in ["mps","auto"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif self.device_asked in ["cpu","auto"]:
            self.device = torch.device("cpu")

        print(f"Device used: {self.device}")
        
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
        self.datasets = DatasetManager(DATA_DIR, self.exp_dir, self.seq_length, self.batch_size, self.num_workers)

        # Model definition
        self.name_weights_to_load = self.args.get("name_weights_to_load", None)

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

        directories = [os.path.join(LOG_DIR,self.exp_dir),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name, 'model_weights'),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name, 'logs')]
        
        self.print_line()
        print("Creating directories...")
        for d in directories:
            os.makedirs(d, exist_ok=True)
            print(f"Directory created (or already existing): {d}")

        save_dir = os.path.join('runs',self.exp_dir, self.exp_name)
        save_path = os.path.join(save_dir, "config.yaml")
        with open(save_path, "w") as f:
            yaml.safe_dump(self.args, f)  # écrit le dictionnaire args dans le fichier
            print(f"Configuration saved at: {save_path}")
        self.print_line()
        



    def execute_experience(self):


        print(f"Starting experiment: {self.exp_name}\n")

        # Directories creation
        self.make_directories()

        # Load or create model
        model = FTO(self.input_dim, self.output_dim, self.representation_dim,
                    self.device, self.modes_separation, self.n_dim,
                    self.domain_size, self.norm_separation, self.seq_length, self.n_heads,
                    self.n_attblocks, self.hidden_dim)
        
        param_dict = model.count_parameters_per_module()
        self.print_line()
        print("Model parameters per module:")
        for name, num in sorted(param_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s}: {num:,} params")
        self.print_line()

        if self.name_weights_to_load is not None:
            path_model = os.path.join('runs',self.exp_dir,self.exp_name,'model_weights')
            loaded_weights = torch.load(os.path.join(path_model, self.name_weights_to_load))
            print(f"Loading weights from {self.name_weights_to_load}, epoch {loaded_weights['epoch']}")
            self.last_epoch = loaded_weights['epoch']
            state_dict = loaded_weights['model_state_dict']
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.\n")

        model = model.to(self.device).float()

        self.optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(), lr=self.optimizer_info["lr"])
        self.scheduler = Factory.get_scheduler(self.scheduler_info, self.optimizer, self.num_epochs, self.datasets.n_batch_train)
        self.metrics = {metric: Factory.get_metric(metric) for metric in self.metrics_name}
        self.loss_fn = Factory.get_metric(self.loss_fn)
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
                          metrics=self.metrics,
                          start_epoch=self.last_epoch if self.name_weights_to_load is not None else 0)
        
        trainer.train_loop()

    def print_line(self):
        print("-------------------------------------------------------")


