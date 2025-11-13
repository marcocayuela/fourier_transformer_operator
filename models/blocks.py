import torch.nn as nn
import torch
import math
import numpy as np


###################################################################################################
class LiftingLayer(nn.Module):

    """Lifting Layer to map input data to higher dimensional space.
    Args:
        input_dim (int): dimension of the input data;
        output_dim (int): dimension of the output data;
        device (str, optional): device to run the model on. Defaults to "cpu".
    Returns:
        nn.Module: Lifting Layer model.
    """    

    def __init__(self, input_dim, output_dim, device="cpu"):
        super(LiftingLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim).to(device)
        self.device = device

    def forward(self, x):
        return self.linear(x)
    
###################################################################################################

class FourierBining():
    """Fourier Bining Layer to separate Fourier modes into different bins.
    Args:
        modes_separation (list): list of integers defining the separation of modes;
        domain_size (int): size of the domain;
        device (str, optional): device to run the model on. Defaults to "cpu".
    Returns:
        FourierBining class: Fourier Bining Layer model.
    """

    def __init__(self, modes_separation, n_dim, domain_size=None, norm="L2", device="cpu"):
        super(FourierBining, self).__init__()
        self.modes_separation = modes_separation
        self.freq_max = self.modes_separation[-1]
        self.n_dim = n_dim
        self.domain_size = n_dim*[1.] if domain_size is None else domain_size
        self.norm = norm
        self.device = device

    def magnitude_spectrum(self):
        k_axes = []
        for L in self.domain_size:
            N = self.freq_max*2
            dx = L/N
            k = np.fft.fftfreq(N, d=dx)
            k_axes.append(k)

        self.k_mesh = np.meshgrid(*k_axes, indexing='ij')
        self.k_magnitude = self.compute_freq_magnitude(self.k_mesh) 
        self.rk_magnitude = self.k_magnitude[...,:self.freq_max+1]

    def compute_freq_magnitude(self, k_mesh):
        if self.norm=="L2":
            k_magnitude = np.sqrt(np.sum([k**2 for k in k_mesh], axis=0)) 
        elif self.norm=="L1":
            k_magnitude = np.sum([np.abs(k) for k in k_mesh], axis=0) 
        elif self.norm=="max":
            k_magnitude = np.max([np.abs(k) for k in k_mesh], axis=0)
        else:
            raise NotImplementedError(f"Norm {self.norm} not implemented.")
        return k_magnitude

    def create_masks(self):
        self.masks = []
        for i in range(len(self.modes_separation)):
            if i==0:
                k_min = 0
            else:
                k_min = self.modes_separation[i-1]
            k_max = self.modes_separation[i]
            mask = np.logical_and(self.rk_magnitude>=k_min, self.rk_magnitude<k_max)
            self.masks.append(mask)

###################################################################################################

class Multihead_Attention(torch.nn.Module):
    """
    Multihead Self-Attention Module.
    """

    def __init__(self, dim: int, num_heads: int, seq_len: int):
        """
        Initialise Multihead_Attention module.

        Parameters:
        - dim       (int) : Dimension of input.
        - num_heads (int) : Number of attention heads.
        - seq_len   (int) : Length of input sequence.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.inv_sqrt_head_dim = 1 / self.head_dim ** 0.5

        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)

        self.softmax = torch.nn.Softmax(dim=-1)

        mask_shape = torch.ones((seq_len, seq_len))
        self.register_buffer("mask", torch.triu(-float("inf") * mask_shape, diagonal=1))

    def forward(self, x):
        
        b = x.size(0)

        q = self.q_proj(x).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(x).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        a = self.softmax(q @ k * self.inv_sqrt_head_dim)

        o = a @ v

        o = self.o_proj(o.transpose(1, 2).reshape(b, -1, self.dim))

        return o, a.mean(1)


###################################################################################################


class Multihead_CrossAttention(torch.nn.Module):
    """
    Multihead Cross-Attention Module.
    """

    def __init__(self, dim: int, num_heads: int, seq_len: int):
        """
        Initialise Multihead_CrossAttention module.

        Parameters:
        - dim       (int) : Dimension of input.
        - num_heads (int) : Number of attention heads.
        - seq_len   (int) : Length of input sequence.
        """
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.inv_sqrt_head_dim = 1 / self.head_dim ** 0.5

        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.o_proj = torch.nn.Linear(dim, dim)

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, c):
        b = x.size(0)

        q = self.q_proj(x).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(c).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(c).reshape(b, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        a = self.softmax(q @ k * self.inv_sqrt_head_dim)
        o = a @ v

        o = self.o_proj(o.transpose(1, 2).reshape(b, -1, self.dim))

        return o, a.mean(1)

###################################################################################################

class Attention_Block(torch.nn.Module):
    """
    Attention Block Module.
    """

    def __init__(self, dim: int, num_heads: int, seq_len: int):
        """
        Initialise Attention_Block module.

        Parameters:
        - dim       (int) : Dimension of input.
        - num_heads (int) : Number of attention heads.
        - seq_len   (int) : Length of input sequence.
        """
        super().__init__()

        self.self_attention = Multihead_Attention(dim, num_heads, seq_len)
        self.cross_attention = Multihead_CrossAttention(dim, num_heads, seq_len)

        self.ln_1 = torch.nn.LayerNorm(dim)
        self.ln_2 = torch.nn.LayerNorm(dim)
        self.ln_3 = torch.nn.LayerNorm(dim)
        self.ln_c = torch.nn.LayerNorm(dim)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim * 2, dim)
        )

    def forward(self, x, c):

        o, a = self.self_attention(self.ln_1(x))
        x = x + o

        o, a = self.cross_attention(self.ln_2(x), self.ln_c(c))
        x = x + o

        o = self.mlp(self.ln_3(x))
        x = x + o

        return x, a

###################################################################################################

class TransformerModel(torch.nn.Module):
    """
    Transformer Module with Cross-Attention.
    """

    def __init__(self, args, model_eval=False):
        """
        Initialise Transformer module.

        Parameters:
        - args      (dict): Model arguments.
        - model_eval (bool): Evaluation mode flag.
        """
        super(TransformerModel, self).__init__()
        self.args = args
        print(f"Transformer {self.args['nattblocks']} attention block(s), {self.args['hidden_dim']} hidden dimensions, {self.args['nheads']} head(s) w Cross attention ")

        self.device = self.args["device"]
        self.dim = self.args["num_obs"]
        self.seq_len = self.args["seq_len"] - 1
        self.num_heads = self.args["nheads"]
        self.nattblocks = self.args['nattblocks']
        self.hidden_dim = self.args['hidden_dim']
        self.param_dim = self.args['nbr_ext_var']

        self.input_embedding = nn.Linear(self.dim, self.hidden_dim)
        self.condition_embedding = nn.Linear(self.param_dim, self.hidden_dim)
        self.output_embedding = nn.Linear(self.hidden_dim, self.dim)

        position = torch.arange(0, self.seq_len).unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000.0) / self.hidden_dim))
        TE = torch.zeros(self.seq_len, self.hidden_dim)
        TE[:, 0::2] = torch.sin(position * div_term)
        TE[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('TE', TE)

        self.att_blocks = torch.nn.ModuleList([
            Attention_Block(self.hidden_dim, self.num_heads, self.seq_len) for _ in range(self.nattblocks)
        ])

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

    def forward(self, z, c):

        z = self.input_embedding(z)
        z = z + self.TE

        c = self.condition_embedding(c)

        for i, att_block in enumerate(self.att_blocks):
            z, a = att_block(z, c)

        z = self.mlp(z[:, -1])
        z = self.output_embedding(z)

        return z
