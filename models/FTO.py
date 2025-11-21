import torch
import torch.nn as nn

from models.blocks import *


class FTO(nn.Module):
    """
    Fourier Transformer Operator model.
    """

    def __init__(self, input_dim, output_dim, representation_dim, device, modes_separation,
                 n_dim, domain_size, norm_separation, seq_len, n_heads, n_attblocks, hidden_dim):
        """
        Initialise FTO module.

        Parameters:
        - args : Arguments containing model hyperparameters.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.representation_dim = representation_dim
        self.device = device
        self.modes_separation = modes_separation
        self.n_dim = n_dim
        self.domain_size = domain_size
        self.norm_separation = norm_separation

        self.seq_len = seq_len
        self.n_heads = n_heads
        self.n_attblocks = n_attblocks
        self.hidden_dim = hidden_dim 
        
        
        self.lifting = LiftingLayer(input_dim=self.input_dim, output_dim=self.representation_dim, device=self.device)
        self.fourierbining = FourierBining(modes_separation=self.modes_separation, n_dim=self.n_dim,
                                            domain_size=self.domain_size, norm_separation=self.norm_separation, device=self.device)
        
        self.transformers = nn.ModuleList([
            TransformerModel(n_obs=self.fourierbining.bins_size[i]*self.representation_dim,
                             seq_len=self.seq_len,
                             n_heads=self.n_heads,
                             hidden_dim=self.hidden_dim,
                             n_attblocks=self.n_attblocks,
                             device=self.device) 
                             for i in range(self.fourierbining.n_bins)])
        
        self.projection = LiftingLayer(input_dim=self.representation_dim, output_dim=self.output_dim, device=self.device)



    def forward(self, x, partial_pred=None):
        x = self.lifting(x)
        x_ft = self.fourierbining.fourier_transform(x)
        binned_x_ft = self.fourierbining.binning(x_ft)
        transformed_bins = []
        for i, bin in enumerate(binned_x_ft):
            if partial_pred is None or i in partial_pred:
                transformer = self.transformers[i]
                transformed_bin = transformer(bin)
            else:
                transformed_bin = torch.zeros_like(bin)
            transformed_bins.append(transformed_bin)
        x_ft_reconstructed = self.fourierbining.unbinning(transformed_bins)
        x = self.fourierbining.inverse_fourier_transform(x_ft_reconstructed)
        x = self.projection(x)
        return x
    

    def predict_sequence(self, x_seq, pred_horizon):
        """
        Predict a sequence of length pred_horizon given input x.

        Parameters:
        - x_seq : Input tensor of shape (batch_size, seq_len, input_dim).
        - pred_horizon : Length of the prediction horizon.

        Returns:
        - predictions : Tensor of shape (batch_size, pred_horizon, output_dim).
        """
        self.eval()
        current_input = x_seq.clone()  # (batch_size, seq_len, dim_1, ..., dim_n, representation_dim)

        for t in range(pred_horizon):
            output = self.forward(current_input) # (batch_size, dim_1, ..., dim_n, representation_dim)
            current_input = torch.cat((current_input[:, 1:, ...], output[:,None,...]), dim=1)  # Slide the window

        return current_input  # (batch_size, pred_horizon, output_dim)
    
    def count_parameters_per_module(self):

        param_dict = {}
        
        for name, module in self.named_children():
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_dict[name] = trainable
        
        param_dict["total"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return param_dict
