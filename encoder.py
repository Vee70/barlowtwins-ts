import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    """ Input:  (N, C_in, L)
        Output: (N, C_out, L)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
    ):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        ))

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalConvBlock(nn.Module):
    """ Input:  (N, C_in, L)
        Output: (N, C_out, L)
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        activation,
        dropout=0.0,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            CausalConv1d(in_channels, out_channels, kernel_size, dilation),
            activation,
            nn.Dropout(dropout),
            CausalConv1d(out_channels, out_channels, kernel_size, dilation),
            activation,
            nn.Dropout(dropout),
        )

        # ensure that input having the same size for residual connection
        self.shortcut = nn.Conv1d(
            in_channels, out_channels, kernel_size=1,
        ) if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.shortcut is None else self.shortcut(x)
        return self.conv_block(x) + res


class CausalConvEncoder(nn.Module):
    """ Input:  (N, L, C_in)
        Output: (N, L, C_out)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        kernel_size,
        n_blocks,
        activation,
        dropout=0.1,
        max_dilation=None,
    ):
        super().__init__()

        act_func = {
            "elu": nn.ELU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(),
            "relu": nn.ReLU(),
        }[activation]

        self.conv_blocks = nn.Sequential(*[
            CausalConvBlock(
                in_dim if i == 0 else hidden_dim,
                hidden_dim if i < n_blocks - 1 else out_dim,
                kernel_size=kernel_size,
                dilation=2**i if max_dilation is None else 2**(i%int(math.log2(max_dilation)+1)),
                activation=act_func,
                dropout=dropout,
            ) for i in range(n_blocks)
        ])

    def forward(self, x):
        return self.conv_blocks(x.transpose(1, 2)).transpose(1, 2)


class MLPHead(nn.Module):
    """ Input:  (N, C_in)
        Output: (N, C_out)
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        bias=False,
        b_norm=True,
    ):
        super().__init__()

        n_block = len(hidden_dim)

        mlp = []
        for i in range(n_block):
            _in_dim = in_dim if i == 0 else hidden_dim[i-1]
            _out_dim = hidden_dim[i]
            # mlp block
            mlp += [ nn.Linear(_in_dim, _out_dim, bias=bias) ]
            if b_norm: mlp += [ nn.BatchNorm1d(_out_dim, affine=False) ]
            mlp += [ nn.ReLU() ]

        # last layer
        mlp += [nn.Linear(hidden_dim[-1], out_dim, bias=bias)]
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x):
        return self.mlp(x)
