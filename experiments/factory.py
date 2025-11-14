import torch
import math

def relative_rmse(y_pred, y_true, eps=1e-8):
    """
    Computes Relative Root Mean Squared Error.
    Args:
        y_pred: Tensor of predictions
        y_true: Tensor of ground truth
        eps: small value to avoid division by zero
    Returns:
        scalar tensor
    """
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    norm = torch.sqrt(torch.mean(y_true ** 2)) + eps
    return rmse / norm

def relative_mae(y_pred, y_true, eps=1e-8):
    """
    Computes Relative Root Mean Squared Error.
    Args:
        y_pred: Tensor of predictions
        y_true: Tensor of ground truth
        eps: small value to avoid division by zero
    Returns:
        scalar tensor
    """
    mae = torch.mean(torch.abs(y_pred - y_true))
    norm = torch.mean(torch.abs(y_true)) + eps
    return mae / norm

class Factory():

    OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    SCHEDULER = {torch.optim.lr_scheduler.OneCycleLR}

    METRICS = {"mse": torch.nn.MSELoss(),
               "rmse": lambda y_pred, y_true: torch.sqrt(torch.mean((y_pred - y_true)**2)),
               "mae": lambda y_pred, y_true: torch.mean(torch.abs(y_pred - y_true)),
               "relative_rmse": relative_rmse,
               "relative_mae": relative_mae}
    
    @staticmethod
    def get_optimizer(name, params, **kwargs):
        return Factory.OPTIMIZERS[name](params, **kwargs)
    
    @staticmethod
    def get_scheduler(name, optimizer, max_lr, n_epoch, n_batch, **kwargs):
        return Factory.SCHEDULERS[name](optimizer, max_lr, n_epoch, math.ceil(n_epoch/n_batch)**kwargs)

    @staticmethod
    def get_metric(name):
        return Factory.METRICS[name]
    