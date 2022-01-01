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
        pretrained=False): # maybe can incorporate pretrained model in here?

        super().__init__()

        if isinstance(activation, str):
            activation = eval('nn.'+activation+'()')

        norm = lambda x: nn.utils.spectral_norm(x) if spectral_norm else x

        # input
        self.layers = nn.ModuleList()
        self.layers.append(norm(nn.Linear(in_dim, n_hidden_units)))
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

class MLP_pretrained(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        # add layers
        self.hidden1 = nn.Linear(2,80)
        self.hidden2 = nn.Linear(80,80)
        self.hidden3 = nn.Linear(80,80)
        self.hidden4 = nn.Linear(80,40)
        self.out = nn.Linear(40,1)
        self.tn = nn.Tanh()
        self.layers = nn.ModuleList([self.hidden1, self.hidden2, self.hidden3, self.hidden4, self.out])
        
    def forward(self, x):
        # do forward pass
        x = self.hidden1(x)
        for l in self.layers[1:]:
            x = l(self.tn(x))
        return x