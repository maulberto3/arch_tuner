from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
from torch import optim
from torch.utils.data import DataLoader, Dataset
from statistics import mean
from torch import load, max as pt_max, ones, save, no_grad, stack, numel, tensor, manual_seed, sigmoid, tanh, add, mul, sub, div
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
from pandas import DataFrame, concat
from pathlib import Path
from string import ascii_lowercase

# Ever wondered how intricate a net can be?

# I coded a dynamic net in the sense that each time it's created, 
# a new random dense architecture is created, probably never-before-seen

# Pretty much I did this only to see if I could discover differences in
# accuracy probably due to interesting net architectures

# check the TODO's below

# https://www.kaggle.com/maulberto3/an-experiment-on-dynamic-nets

# manual_seed(16)
# seed(16)

class ARCH_NET(nn.Module):
    def __init__(
        self,
        input_size=784,
        hidden_dim=10,
        output_dim=10,
        nodes=5
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.nodes = nodes
        # self.activs = [F.selu, F.relu, F.celu, F.elu, sigmoid, F.logsigmoid, F.softplus, F.softsign, tanh]
        self.activs = [F.elu, sigmoid, F.softsign]
        self.opers = [add, mul, sub]

        self.abc = [letter*j for j in range(1, 99) for letter in ascii_lowercase]

        self.init_layers = nn.ModuleDict()  # required for optim
        self.init_activs = OrderedDict()

        self.hidden_layers = nn.ModuleDict()
        self.hidden_activs = OrderedDict()
        self.helper = OrderedDict()
        self.helper2 = OrderedDict()
        self.hidden_opers = OrderedDict()
        self.helper3 = OrderedDict()

        self.to_output = nn.Linear(hidden_dim, output_dim)
        
    def create_initial_layer(self):
        """Creates the NN initial layer"""
        for _ in range(2):
            self.init_layers['i' + str(len(self.init_layers))] = nn.Linear(self.input_size, self.hidden_dim)
            self.init_activs['i' + str(len(self.init_activs))] = choice(self.activs)

    def create_node(self, options):
        """Create a NN node"""
        pair = list(combinations(options, 2))
        pair = choice(pair)
 
        if pair in self.helper:
            self.create_node(self.options)
        else:
            self.options.append(pair)

            self.helper[pair] = self.abc[len(self.hidden_layers)]
            self.hidden_layers[self.abc[len(self.hidden_layers)]] = nn.Linear(self.hidden_dim, self.hidden_dim)
            
            self.hidden_activs[self.abc[len(self.hidden_layers)]] = choice(self.activs)
            self.helper2[pair] = self.hidden_activs[self.abc[len(self.hidden_layers)]].__name__
            
            self.hidden_opers[self.abc[len(self.hidden_layers)]] = choice(self.opers)
            self.helper3[pair] = self.hidden_opers[self.abc[len(self.hidden_layers)]].__name__

            ##########
            # TODO also put sign into it for merging layers
            ##########
        return pair
            
    def create_random_arch(self):
        """Create a random NN Architecture"""
        self.create_initial_layer()
        for _ in range(self.nodes):
            self.options = list(self.init_layers.keys())
            self.options.extend(list(self.helper.keys()))
            self.create_node(self.options)
        # print('Net Architecture is')
        # pprint(self.helper)

        # def create_arch_from(self, arch):
        # """Creates a user-supplied NN Architecture"""
        
    def forward(self, x):
        """Forward/Predict"""
        init_l = {}
        for (lay_k, lay), (act) in zip(
            self.init_layers.items(),
            self.init_activs.values()):
            init_l[lay_k] = act(F.dropout((lay(x)), p=0.2))
        
        hidden_l = {}
        for (pair), (lay), (act), (oper) in zip(
                self.helper.keys(),
                self.hidden_layers.values(),
                self.hidden_activs.values(),
                self.hidden_opers.values()):
            hidden_l[pair] = oper({**init_l, **hidden_l}[pair[0]], {**init_l, **hidden_l}[pair[1]])
            hidden_l[pair] = act(F.dropout(lay(hidden_l[pair]), p=0.2))
        
        hidden_l = list(hidden_l.values())[-1]
        hidden_l = self.to_output(hidden_l)
        hidden_l = F.log_softmax(hidden_l, dim=-1).squeeze()
        return hidden_l
    

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
        return X.float(), y.long()

# HP
BATCH_SIZE = 128
LR = 0.0005

# Data
train = load(Path.home() / 'data' / 'MNIST' / 'processed' / 'training.pt')
test = load(Path.home() / 'data' / 'MNIST' / 'processed' / 'test.pt')

archs = []

for i in range(4):
    print(f'{i:^3}', end=' ')
    # Net
    net = ARCH_NET(hidden_dim=64, nodes=4).train()
    net.create_random_arch()
    net = net.cuda()

    # print(f'{sum([numel(sublayer) for layer in net.init_layers.values() for sublayer in layer.parameters()]):,}')
    # print(f'{sum([numel(sublayer) for layer in net.hidden_layers.values() for sublayer in layer.parameters()]):,}')
    # print(list(net.named_parameters()))
    
    optimizer = Adam(net.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    # Train
    train_ds = DS(train[0], train[1])
    train_dl = DataLoader(train_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        drop_last=True)
    for epoch in range(1):
        #running_loss = []
        train_correct = 0
        train_total = 0
        for inputs, labels in train_dl:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, labels).mean()
            loss.backward()
            optimizer.step()

            # print statistics
            #running_loss.append(loss.item())
            _, predicted = pt_max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        print(f'TRAIN {train_correct / train_total * 100:^5.2f} %', end=' ')
        
        # archs.append([tuple(net.helper), 100 * train_correct / train_total])
        
    # Eval
    with no_grad():
        net = net.eval()
        test_ds = DS(test[0], test[1])
        test_dl = DataLoader(test_ds,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            drop_last=True)
        #running_loss = []
        test_correct = 0
        test_total = 0
        for inputs, labels in test_dl:
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = net(inputs)

            # print statistics
            #running_loss.append(loss.item())
            _, predicted = pt_max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
        print(f'TEST {test_correct / test_total * 100:^5.2f} %')
            
    archs.extend(list(zip(
        net.helper.values(),
        net.helper.keys(),
        net.helper3.values(), 
        net.helper2.values(),
        repeat(100 * test_correct / test_total))))

t = DataFrame(archs)
t[0] = t[0].astype(str)
# print(t)
print(t.shape)
t.to_csv('temp.csv', index=False)

# print(t2.groupby(2).mean().squeeze())
# print(t2.groupby(2).std().squeeze())

# print(t2.groupby(6).mean().squeeze())
# print(t2.groupby(6).std().squeeze())