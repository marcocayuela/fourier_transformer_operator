import csv
import os 
import inspect
import yaml
import time 
from tqdm import tqdm

import pandas as pd
import torch
import numpy as np

from models.FTO import FTO
from experiments.factory import Factory
from training.dataset_manager import DatasetManager

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

##################### Which experiment to test ##########################
exp_dir = 'kolmogorov'
exp_name = 'exp1'
model_to_load = 'min_test_loss.pth'

n_epoch = 100
#########################################################################

LOG_DIR = os.getenv("LOG_DIR", "./runs")  
DATA_DIR = os.getenv("DATA_DIR", "./data")  

if __name__ == "__main__":

    if not(os.path.exists(os.path.join(LOG_DIR,exp_dir,exp_name))):
        raise ValueError(f"Experiment {exp_name} in directory {exp_dir} does not exist.")
    else:
        print(f"Generating plots for experiment {exp_name} in directory {exp_dir}...")
        metrics = pd.read_csv(open(os.path.join(LOG_DIR,exp_dir, exp_name, 'logs', 'metrics.csv')))

        metrics = metrics[-n_epoch:]

    

        ################## Plotting Loss ##############################
        fig, ax1 = plt.subplots(figsize=(8,5))

        # Train/Test loss curves
        ax1.plot(metrics['Epoch'], metrics['Tr loss'], label='Train loss', color='#1f77b4', linewidth=2, linestyle='--')
        ax1.plot(metrics['Epoch'], metrics['Te loss'], label='Test loss', color='#ff7f0e', linewidth=2, linestyle='--')
        ax1.set_ylabel('Losses')
        ax1.tick_params(axis='y')
        ax1.set_xlabel('Epoch')

        # LR curve
        ax2 = ax1.twinx()
        ax2.plot(metrics['Epoch'], metrics['LR'], color='#2ca02c', label='Learning Rate', linewidth=2)
        ax2.set_ylabel('Learning Rate', color='#2ca02c')
        ax2.tick_params(axis='y', colors='#2ca02c')
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))

        # Légende combinée
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

        plt.savefig(os.path.join(LOG_DIR,exp_dir, exp_name, 'logs', 'loss_and_lr_plot.png'), dpi=300)
        print(f"Plot saved at: {os.path.join(LOG_DIR,exp_dir, exp_name, 'logs', 'loss_and_lr_plot.png')}\n")
        plt.close()

        #####################################################################
        ################## Load experiment config ###########################
        path_config = os.path.join(LOG_DIR, exp_dir, exp_name, 'config.yaml')
        with open(path_config, "r") as f:
            args = yaml.safe_load(f)

        #Device parameters
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        args["device"] = device

        #####################################################################
        ################## Load model weights ###############################
        sig = inspect.signature(FTO.__init__)
        accepted_params = sig.parameters.keys() 
        filtered_args = {k: v for k, v in args.items() if k in accepted_params and k != 'self'}

        model = FTO(**filtered_args)

        path_model = os.path.join(LOG_DIR, exp_dir, exp_name, 'model_weights', model_to_load)
        loaded_weights = torch.load(os.path.join(path_model))
        print(f"Loading weights from {path_model}, epoch {loaded_weights['epoch']}")
        last_epoch = loaded_weights['epoch']
        state_dict = loaded_weights['model_state_dict']
        model.load_state_dict(state_dict)
        print("Weights loaded successfully.\n")

        ######################################################################
        ################## Prepare metrics ###################################
        metrics = {metric: Factory.get_metric(metric) for metric in args["metrics"]}
        loss_fn = Factory.get_metric(args["loss_fn"])
        metrics['loss'] = loss_fn

        ######################################################################
        ################## Load datasets #####################################
        datasets = DatasetManager(DATA_DIR, exp_dir, args["seq_length"], args["batch_size"], args["num_workers"])

        ######################################################################
        ################## Evaluate model ####################################

        ### How it was train (predict sequences)
        model = model.to(device).float()
        model.eval()
        # with torch.no_grad():
        #     test_loss = 0.0
        #     test_metrics = {k: 0.0 for k in metrics.keys()}

        #     print("Evaluating model with sequence prediction on test set...")
        #     for batch_idx, (inputs, targets) in enumerate(tqdm(datasets.testing_loader)):
        #         inputs, targets = inputs.to(device).float(), targets.to(device).float()

        #         outputs = model.predict_sequence(inputs, pred_horizon=targets.shape[1])
        #         loss = loss_fn(outputs, targets)
        #         test_loss += loss.item()

        #         for name, metric_fn in metrics.items():
        #             test_metrics[name] += metric_fn(outputs, targets).item()

            
        #     avg_test_loss = test_loss / len(datasets.testing_loader)
        #     for name in test_metrics.keys():
        #         test_metrics[name] /= len(datasets.testing_loader)
        #     test_metrics["loss"] = avg_test_loss

        #     metrics_results = {loss_name: f"{loss_value:.6f}" for loss_name, loss_value in test_metrics.items()}
        #     metrics_results["loss"] = f"{avg_test_loss:.6f}"
        #     print("Test set metrics:")
        #     for name, value in metrics_results.items():
        #         print(f"{name:15s}: {value}")
        #     save_path = os.path.join('runs', exp_dir, exp_name, 'logs', 'final_test_metrics.txt')
        #     with open(save_path, 'w') as f:
        #         f.write("Test set metrics:\n")
        #         for name, value in metrics_results.items():
        #             f.write(f"{name:15s}: {value}\n")
        #     print(f"\nMetrics saved at: {save_path}")

        #     train_loss = 0.0
        #     train_metrics = {k: 0.0 for k in metrics.keys()}

        #     print("Evaluating model with sequence prediction on train set...")
        #     for batch_idx, (inputs, targets) in enumerate(tqdm(datasets.training_loader)):
        #         inputs, targets = inputs.to(device).float(), targets.to(device).float()

        #         outputs = model.predict_sequence(inputs, pred_horizon=targets.shape[1])
        #         loss = loss_fn(outputs, targets)
        #         train_loss += loss.item()

        #         for name, metric_fn in metrics.items():
        #             train_metrics[name] += metric_fn(outputs, targets).item()

            
        #     avg_train_loss = train_loss / len(datasets.training_loader)
        #     for name in train_metrics.keys():
        #         train_metrics[name] /= len(datasets.training_loader)
        #     train_metrics["loss"] = avg_train_loss

        #     metrics_results = {loss_name: f"{loss_value:.6f}" for loss_name, loss_value in train_metrics.items()}
        #     metrics_results["loss"] = f"{avg_train_loss:.6f}"
        #     print("train set metrics:")
        #     for name, value in metrics_results.items():
        #         print(f"{name:15s}: {value}")
        #     save_path = os.path.join('runs', exp_dir, exp_name, 'logs', 'final_train_metrics.txt')
        #     with open(save_path, 'w') as f:
        #         f.write("train set metrics:\n")
        #         for name, value in metrics_results.items():
        #             f.write(f"{name:15s}: {value}\n")
        #     print(f"\nMetrics saved at: {save_path}")


        # # unique snapshot prediction
        # with torch.no_grad():
        #     test_loss = 0.0
        #     test_metrics = {k: 0.0 for k in metrics.keys()}

        #     print("Evaluating model with single step prediction on test set...")
        #     for batch_idx, (inputs, targets) in enumerate(tqdm(datasets.testing_loader)):
        #         inputs, targets = inputs.to(device).float(), targets.to(device).float()

        #         outputs = model(inputs)
        #         loss = loss_fn(outputs, targets[:,0,...])  # only first step target
        #         test_loss += loss.item()

        #         for name, metric_fn in metrics.items():
        #             test_metrics[name] += metric_fn(outputs, targets[:,0,...]).item()

            
        #     avg_test_loss = test_loss / len(datasets.testing_loader)
        #     for name in test_metrics.keys():
        #         test_metrics[name] /= len(datasets.testing_loader)
        #     test_metrics["loss"] = avg_test_loss

        #     metrics_results = {loss_name: f"{loss_value:.6f}" for loss_name, loss_value in test_metrics.items()}
        #     metrics_results["loss"] = f"{avg_test_loss:.6f}"
        #     print("Test set metrics:")
        #     for name, value in metrics_results.items():
        #         print(f"{name:15s}: {value}")
        #     save_path = os.path.join('runs', exp_dir, exp_name, 'logs', 'final_test_metrics.txt')
        #     with open(save_path, 'w') as f:
        #         f.write("Test set metrics:\n")
        #         for name, value in metrics_results.items():
        #             f.write(f"{name:15s}: {value}\n")
        #     print(f"\nMetrics saved at: {save_path}")

        #     train_loss = 0.0
        #     train_metrics = {k: 0.0 for k in metrics.keys()}

        #     print("Evaluating model with single step prediction on train set...")
        #     for batch_idx, (inputs, targets) in enumerate(tqdm(datasets.training_loader)):
        #         inputs, targets = inputs.to(device).float(), targets.to(device).float()

        #         outputs = model(inputs)
        #         loss = loss_fn(outputs, targets[:,0,...])  # only first step target
        #         train_loss += loss.item()

        #         for name, metric_fn in metrics.items():
        #             train_metrics[name] += metric_fn(outputs, targets[:,0,...]).item()

            
        #     avg_train_loss = train_loss / len(datasets.training_loader)
        #     for name in train_metrics.keys():
        #         train_metrics[name] /= len(datasets.training_loader)
        #     train_metrics["loss"] = avg_train_loss

        #     metrics_results = {loss_name: f"{loss_value:.6f}" for loss_name, loss_value in train_metrics.items()}
        #     metrics_results["loss"] = f"{avg_train_loss:.6f}"
        #     print("train set metrics:")
        #     for name, value in metrics_results.items():
        #         print(f"{name:15s}: {value}")
        #     save_path = os.path.join('runs', exp_dir, exp_name, 'logs', 'final_train_metrics.txt')
        #     with open(save_path, 'w') as f:
        #         f.write("train set metrics:\n")
        #         for name, value in metrics_results.items():
        #             f.write(f"{name:15s}: {value}\n")
        #     print(f"\nMetrics saved at: {save_path}")


        # Fully autoregressive prediction
        with torch.no_grad():
            test_loss = 0.0
            test_metrics = {k: 0.0 for k in metrics.keys()}

            print("Evaluating model on the whole test set with autoregressive prediction...")

            target = torch.tensor(datasets.testing_set[args["seq_length"]:])
            input_seq = torch.tensor(datasets.testing_set[:args["seq_length"]]).unsqueeze(0)
            input_seq, target = input_seq.float().to(device), target.float().to(device)

            autoregressive_prediction = model.predict_sequence(input_seq, pred_horizon=target.shape[0], barplot=True).squeeze()

            loss = loss_fn(autoregressive_prediction, target)
            test_loss += loss.item()

            for name, metric_fn in metrics.items():
                test_metrics[name] += metric_fn(autoregressive_prediction, target).item()
            test_metrics["loss"] = test_loss

            metrics_results = {loss_name: f"{loss_value:.6f}" for loss_name, loss_value in test_metrics.items()}
            metrics_results["loss"] = f"{test_loss:.6f}"
            print("Test set metrics:")
            for name, value in metrics_results.items():
                print(f"{name:15s}: {value}")
            save_path = os.path.join(LOG_DIR, exp_dir, exp_name, 'logs', 'final_test_metrics.txt')
            with open(save_path, 'w') as f:
                f.write("Test set metrics:\n")
                for name, value in metrics_results.items():
                    f.write(f"{name:15s}: {value}\n")
            print(f"\nMetrics saved at: {save_path}")

            train_loss = 0.0
            train_metrics = {k: 0.0 for k in metrics.keys()}

            print("Evaluating model on the whole train set with autoregressive prediction...")

            target = torch.tensor(datasets.training_set[args["seq_length"]:])
            input_seq = torch.tensor(datasets.training_set[:args["seq_length"]]).unsqueeze(0)
            input_seq, target = input_seq.float().to(device), target.float().to(device)
            
            autoregressive_prediction = model.predict_sequence(input_seq, pred_horizon=target.shape[0], barplot=True).squeeze()
            loss = loss_fn(autoregressive_prediction, target)
            train_loss += loss.item()

            for name, metric_fn in metrics.items():
                train_metrics[name] += metric_fn(autoregressive_prediction, target).item()
            train_metrics["loss"] = train_loss

            metrics_results = {loss_name: f"{loss_value:.6f}" for loss_name, loss_value in train_metrics.items()}
            metrics_results["loss"] = f"{train_loss:.6f}"
            print("train set metrics:")
            for name, value in metrics_results.items():
                print(f"{name:15s}: {value}")
            save_path = os.path.join(LOG_DIR, exp_dir, exp_name, 'logs', 'final_train_metrics.txt')
            with open(save_path, 'w') as f:
                f.write("train set metrics:\n")
                for name, value in metrics_results.items():
                    f.write(f"{name:15s}: {value}\n")
            print(f"\nMetrics saved at: {save_path}")


    
    predictions = np.concatenate([input_seq.squeeze().cpu().numpy(), autoregressive_prediction.cpu().numpy()], axis=0)
    target = datasets.training_set
    print(predictions.shape, target.shape)
    relative_error_time = np.sqrt(np.sum((predictions - target)**2, axis=(1,2,3)) / np.sum(target**2, axis=(1,2,3)))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(np.arange(relative_error_time.shape[0]), relative_error_time, label='Relative Error over Time', color='blue', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Relative Error', color='blue')
    ax.tick_params(axis='y', colors='blue') 
    plt.title('Relative Error of Autoregressive Prediction Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(LOG_DIR,exp_dir, exp_name, 'logs', 'relative_error_over_time.png'), dpi=300)
    print(f"Plot saved at: {os.path.join(LOG_DIR,exp_dir, exp_name, 'logs', 'relative_error_over_time.png')}\n")
    plt.close()


    cinetic_energy_pred = 0.5 * np.mean(np.sum(predictions**2, axis=3), axis=(1,2))
    cinetic_energy_true = 0.5 * np.mean(np.sum(target**2, axis=3), axis=(1,2))

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(np.arange(cinetic_energy_true.shape[0]), cinetic_energy_true, label='True Cinetic Energy', color='green', linewidth=2)
    ax.plot(np.arange(cinetic_energy_pred.shape[0]), cinetic_energy_pred, label='Predicted Cinetic Energy', color='orange', linewidth=2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Cinetic Energy')
    plt.title('Cinetic Energy Over Time')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(LOG_DIR,exp_dir, exp_name, 'logs', 'cinetic_energy_over_time.png'), dpi=300)
    print(f"Plot saved at: {os.path.join(LOG_DIR,exp_dir, exp_name, 'logs', 'cinetic_energy_over_time.png')}\n")
    plt.close()
