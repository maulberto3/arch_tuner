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

manual_seed(16)
seed(16)

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
        
        self.simple_arch = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.Dropout(p=0.2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=0.2),
            nn.Softsign(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(p=0.2),
            # nn.Softsign(),
            nn.Linear(hidden_dim, output_dim),
            nn.LogSoftmax(dim=-1))
        
    def forward(self, x):
        x = self.simple_arch(x).squeeze(1)
        # print(x.shape)
        return x
    
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

for _ in range(1):
    # Net
    net = ARCH_NET(hidden_dim=256, nodes=10).train()
    # net.create_random_arch()
    net = net.cuda()

    #f'{sum([numel(sublayer) for layer in net.init_layers.values() for sublayer in list(layer.parameters())]):,}'
    #f'{sum([numel(sublayer) for layer in net.layers.values() for sublayer in list(layer.parameters())]):,}'
    # list(net.named_parameters())
    
    optimizer = Adam(net.parameters(), lr=LR)
    criterion = nn.NLLLoss()

    # Train
    train_ds = DS(train[0], train[1])
    train_dl = DataLoader(train_ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        drop_last=True)

    for epoch in range(5):
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
            
    # archs.append([list(zip(net.helper, net.helper2.values())), 100 * test_correct / test_total])
