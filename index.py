from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from torch import optim
from torch.utils.data import DataLoader, Dataset
from statistics import mean
from torch import load, max as pt_max, ones, save, no_grad, stack, numel, tensor, manual_seed, sigmoid, tanh
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from optuna import create_study
from itertools import repeat, product, combinations
from collections import defaultdict, OrderedDict
from random import choice, seed
from torchviz import make_dot
from pprint import pprint
import torchvision
import torchvision.transforms as transforms

# Ever wondered how intricate a net can be?

# I coded a dynamic net in the sense that each time it's created, 
# a new random dense architecture is created, probably never-before-seen

# Pretty much I did this only to see if I could discover differences in
# accuracy probably due to interesting net architectures

# check the TODO's below

manual_seed(16)
seed(16)

class ARCH_NET(nn.Module):
    def __init__(
        self,
        input_size=2048,
        hidden_dim=10,
        output_dim=4,
        nodes=5
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.nodes = nodes
        self.activ = [F.selu, F.relu, F.celu, F.elu, sigmoid, F.logsigmoid, F.softplus, F.softsign, tanh]
        
        self.start = 'ab'
        self.options = list(self.start)
        
        self.create_random_arch()
        
        self.to_output = nn.Linear(hidden_dim, output_dim)
        
    def create_initial_layer(self):
        self.init_layers = nn.ModuleDict()
        self.init_activs = {}
        for init in self.start:
            self.init_layers[init] = nn.Linear(self.input_size, self.hidden_dim)
            self.init_activs[init] = choice(self.activ)

    def create_node(self, options):
        pair = list(combinations(options, 2))
        pair = choice(pair)
        #pair =  "".join(pair)
        if pair in self.helper:
            pair = self.create_node(options)
        else:
            self.helper[pair] = str(len(self.hidden_layers))
            self.hidden_layers[str(len(self.hidden_layers))] = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.hidden_activs[str(len(self.hidden_layers))] = choice(self.activ)
            self.helper2[pair] = self.hidden_activs[str(len(self.hidden_layers))].__name__
        return pair
            
    def create_random_arch(self):
        self.create_initial_layer()
        self.helper = {}
        self.helper2 = {}
        self.hidden_layers = nn.ModuleDict()
        self.hidden_activs = {}
        options = self.options.copy()
        for i in range(self.nodes):
            res = self.create_node(options)
            options.append(res)
        #print('Net Architecture is')
        #pprint(list(zip(
         #       self.init_layers.items(),
          #      self.init_activs.items())))

        # def create_arch_from(self):
        
    def forward(self, x):
        init_l = {lay[0]: act[1](F.dropout(lay[1](x), p=0.4)) for lay, act in zip(
                self.init_layers.items(),
                self.init_activs.items())}
        hidden_l = {}
        for a, b, c  in zip(
                self.helper.items(),
                self.hidden_layers.items(),
                self.hidden_activs.items()):
            hidden_l[a[0]] = {**init_l, **hidden_l}[a[0][0]] + {**init_l, **hidden_l}[a[0][1]]
            hidden_l[a[0]] = c[1](F.dropout(b[1](hidden_l[a[0]]), p=0.4))
        hidden_l = stack(list(hidden_l.values()))
        hidden_l = hidden_l.sum(dim=0)
        hidden_l = self.to_output(hidden_l)
        hidden_l = F.log_softmax(hidden_l, dim=-1).squeeze()
        return hidden_l
    
net = ARCH_NET()
# net
# make_dot(net(maps[0].ravel()))

class DS(Dataset):
    def __init__(self, maps, labels) -> None:
        self.maps = maps
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.maps[idx]
        X = X.reshape(1, -1)
        y = self.labels[idx]
        return X, y

# HP
BATCH_SIZE = 64
LR = 0.0005

# Data
from sklearn.datasets import load_digits
data = load_

# Split
# train_X, test_X, train_y, test_y = train_test_split(maps, labels)