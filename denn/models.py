import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """ Most basic residual block
        https://arxiv.org/pdf/1512.03385.pdf : Equation #1
    """

    def __init__(self, n_units, activation, spectral_norm=False):
        super(ResidualBlock, self).__init__()

        norm = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x

        self.activation = activation
        self.l1 = norm(nn.Linear(n_units, n_units))
        self.l2 = norm(nn.Linear(n_units, n_units))

    def forward(self, x):
        return self.activation(
            self.l2(self.activation(self.l1(x))) + x
        )

class MLP(nn.Module):
    """
    Multi-layer perceptron

    Optionally use residual connections
    Default is sigmoid output
    """
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2,
        activation=nn.Tanh(), residual=False, regress=False, spectral_norm=False, 
        n_heads=1, pretrained=False):

        super().__init__()

        if isinstance(activation, str):
            activation = eval('nn.'+activation+'()')

        norm = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x

        # input
        self.layers = nn.ModuleList()
        self.layers.append(norm(nn.Linear(in_dim*n_heads, n_hidden_units)))
        self.layers.append(activation)

        # hidden
        for l in range(n_hidden_layers):
            if residual:
                self.layers.append(ResidualBlock(n_hidden_units, activation, spectral_norm=spectral_norm))
            else:
                self.layers.append(norm(nn.Linear(n_hidden_units, n_hidden_units)))
                self.layers.append(activation)

        # output
        self.layers.append(norm(nn.Linear(n_hidden_units, out_dim)))
        if not regress:
            # For WGAN: Discriminator should be unbounded (i.e. no sigmoid / regress = True)
            self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x

class MultiHeadGen(nn.Module):
    """
    Multi-layer perceptron

    Optionally use residual connections
    Default is sigmoid output
    """
    def __init__(self, in_dim=1, out_dim=1, n_hidden_units=20, n_hidden_layers=2,
        activation=nn.Tanh(), residual=False, regress=False, spectral_norm=False,
        n_heads=1, n_head_units=20, pretrained=False):

        super().__init__()
        self.n_heads = n_heads
        
        if isinstance(activation, str):
            activation = eval('nn.'+activation+'()')

        # input
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, n_hidden_units))
        self.layers.append(activation)

        # hidden
        for l in range(n_hidden_layers):
            if residual:
                self.layers.append(ResidualBlock(n_hidden_units, activation, spectral_norm=spectral_norm))
            else:
                self.layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                self.layers.append(activation)

        # heads
        # self.heads = nn.ModuleList([nn.ModuleList([norm(nn.Linear(n_hidden_units, n_hidden_units))])]) 
        # self.heads.extend([nn.ModuleList([norm(nn.Linear(n_hidden_units, n_hidden_units))]) for l in range(self.n_heads-1)])
        # for head in self.heads:
        #     head.append(activation)
        #     head.append(norm(nn.Linear(n_hidden_units, n_hidden_units)))
        self.heads = nn.ModuleList([nn.Linear(n_hidden_units, n_head_units)]) 
        self.heads.extend([nn.Linear(n_hidden_units, n_head_units) for l in range(self.n_heads-1)])

        # output
        self.outputs = nn.ModuleList([nn.Linear(n_head_units, out_dim)]) 
        self.outputs.extend([nn.Linear(n_head_units, out_dim) for l in range(self.n_heads-1)])

    def forward(self, x):
        d = {}
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        for j in range(self.n_heads):
            x1 = self.heads[j](x)
            # for k in range(len(self.heads[j])):
            #     x1 = self.heads[j][k](x)
            x1 = self.outputs[j](x1)
            d[j] = x1
        return d

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, input):
        return input * torch.sigmoid(self.beta * input)

class TorchSin(nn.Module):
    """
    Sin activation function
    """
    def __init__(self):
        super(TorchSin, self).__init__()

    def forward(self, x):
        return torch.sin(x)
