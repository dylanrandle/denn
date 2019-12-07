import torch
import torch.nn as nn

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, input):
        return input * torch.sigmoid(self.beta * input)

    def extra_repr(self):
        return 'beta={}'.format(self.beta)

class TorchSin(nn.Module):
    """
    Sin activation function
    """
    def __init__(self):
        super(TorchSin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class ResidualBlock(nn.Module):
    """ Most basic residual block
        https://arxiv.org/pdf/1512.03385.pdf : Equation #1
    """

    def __init__(self, n_units, activation):
        super(ResidualBlock, self).__init__()

        self.activation = activation
        self.l1 = nn.Linear(n_units, n_units)
        self.l2 = nn.Linear(n_units, n_units)

    def forward(self, x):
        return self.activation(
            self.l2(self.activation(self.l1(x))) + x
        )

class MLP(nn.Module):
    """
    Multi-layer perceptron

    Optionally use residual connections, output Tanh, or be unbounded.
    Default is sigmoid output
    """
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2,
        activation=nn.Tanh(), residual=False, regress=False,
        # output_tan=False, # this was added as a hack but I believe is not necessary
        ):

        super().__init__()

        # input
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, n_hidden_units))
        self.layers.append(activation)

        # hidden
        for l in range(n_hidden_layers):
            if residual:
                self.layers.append(ResidualBlock(n_hidden_units, activation))
            else:
                self.layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                self.layers.append(activation)

        # output
        self.layers.append(nn.Linear(n_hidden_units, out_dim))
        if not regress:
            # For WGAN: Discriminator should be unbounded (i.e. no sigmoid / regress = True)
            self.layers.append(nn.Sigmoid())

        # this was added as a hack but I believe is not necessary
        # if output_tan:
        #     self.layers.append(nn.Tanh())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x
