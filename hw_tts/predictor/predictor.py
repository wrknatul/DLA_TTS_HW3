import torch.nn as nn

from collections import OrderedDict


class Transpose(nn.Module):
    """Transpose as nn.Module"""

    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class Predictor(nn.Module):
    """ Duration Predictor """

    def __init__(self,
                 encoder_dim,
                 duration_predictor_filter_size,
                 duration_predictor_kernel_size,
                 dropout=0.1):
        super(Predictor, self).__init__()

        self.input_size = encoder_dim
        self.filter_size = duration_predictor_filter_size
        self.kernel = duration_predictor_kernel_size
        self.conv_output_size = duration_predictor_filter_size
        self.dropout = dropout

        self.conv_layer = nn.ModuleDict(OrderedDict([
            ("transpose1", Transpose(-1, -2)),
            ("conv1d_1", nn.Conv1d(self.input_size,
                                   self.filter_size,
                                   kernel_size=self.kernel,
                                   padding=1)),
            ("transpose2", Transpose(-1, -2)),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("relu_1", nn.ReLU()),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("transpose3", Transpose(-1, -2)),
            ("conv1d_2", nn.Conv1d(self.filter_size,
                                   self.filter_size,
                                   kernel_size=self.kernel,
                                   padding=1)),
            ("transpose4", Transpose(-1, -2)),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("relu_2", nn.ReLU()),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        for key, layer in self.conv_layer.items():
            encoder_output = layer(encoder_output)

        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out
