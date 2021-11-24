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
from pandas import DataFrame, concat
from pathlib import Path

# Ever wondered how intricate a net can be?

# I coded a dynamic net in the sense that each time it's created, 
# a new random dense architecture is created, probably never-before-seen

# Pretty much I did this only to see if I could discover differences in
# accuracy probably due to interesting net architectures

# check the TODO's below

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
        self.activ = [F.selu, F.relu, F.celu, F.elu, sigmoid, F.logsigmoid, F.softplus, F.softsign, tanh]
        
        self.start = 'ab'
        self.options = list(self.start)
        
        self.init_layers = nn.ModuleDict()
        self.init_activs = {}
        self.helper = OrderedDict()
        self.helper2 = OrderedDict()
        self.hidden_layers = nn.ModuleDict()
        self.hidden_activs = {}

        self.create_random_arch()
        
        self.to_output = nn.Linear(hidden_dim, output_dim)
        
    def create_initial_layer(self):
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
        options = self.options.copy()
        for i in range(self.nodes):
            pair = self.create_node(options)
            options.append(pair)
        #print('Net Architecture is')
        #pprint(self)

        # def create_arch_from(self, arch):

        
    def forward(self, x):
        init_l = {lay[0]: act[1](F.dropout(lay[1](x), p=0.2)) for lay, act in zip(
                self.init_layers.items(),
                self.init_activs.items())}
        hidden_l = {}
        for a, b, c  in zip(
                self.helper.items(),
                self.hidden_layers.items(),
                self.hidden_activs.items()):
            hidden_l[a[0]] = {**init_l, **hidden_l}[a[0][0]] * {**init_l, **hidden_l}[a[0][1]]
            hidden_l[a[0]] = c[1](F.dropout(b[1](hidden_l[a[0]]), p=0.2))
        hidden_l = stack(list(hidden_l.values()))
        hidden_l = hidden_l.sum(dim=0)
        hidden_l = self.to_output(hidden_l)
        hidden_l = F.log_softmax(hidden_l, dim=-1).squeeze()
        return hidden_l
    
# net = ARCH_NET()
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
        return X.float(), y.long()

# HP
BATCH_SIZE = 128
LR = 0.0001

# Data
train = load(Path.home() / 'data' / 'MNIST' / 'processed' / 'training.pt')
test = load(Path.home() / 'data' / 'MNIST' / 'processed' / 'test.pt')

archs = []

for i in range(200):
    print(i)
    # Net
    net = ARCH_NET(hidden_dim=64, nodes=6).train()
    net.create_random_arch()
    net = net.cuda()

    # Arch
    # pprint(net.helper)
    # pprint(net.helper2)

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
        print(f'TRAIN {train_correct / train_total * 100:.2f} %')
        
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
        print(f'TEST {test_correct / test_total * 100:.2f} %')
            
    archs.extend(list(zip(net.helper.values(), net.helper2.items(), repeat(100 * test_correct / test_total))))

t = DataFrame(archs)
t1 = t[0].astype(str)
t1 = t1.str.split(',', expand=True)
t2 = concat([t1, t], axis=1)
# print(t)
print(t.shape)
t2.to_csv('temp2.csv', index=False)

# print(t2.groupby(2).mean().squeeze())
# print(t2.groupby(2).std().squeeze())

# print(t2.groupby(6).mean().squeeze())
# print(t2.groupby(6).std().squeeze())