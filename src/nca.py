import pandas as pd
from scipy import sparse
from torch import nn
import torch.nn.functional as f
import torch
from torch.utils.data import TensorDataset, DataLoader

from os.path import join as path_join
import numpy as np

class NCA(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.n_users = config['n_users']
    self.n_items = config['n_items']
    self.k = config['k']

    self.embed_user = nn.Embedding(self.n_users, self.k)
    self.embed_item = nn.Embedding(self.n_items, self.k)

    self.fc_layers = nn.ModuleList()
    for idx, (in_size, out_size) in enumerate(zip(config['layers'][:-1], config['layers'][1:])):
        self.fc_layers.append(torch.nn.Linear(in_size, out_size))


    self.dropout = nn.Dropout(0.2)

    self.output = nn.Linear(config['layers'][-1],  1)
    self.output_f = nn.Sigmoid()

  def forward(self, users, items):

    users_x = self.embed_user(users)
    items_x = self.embed_item(items)

    x = torch.cat([users_x, items_x], dim = 1) # Concatenate along the second axis

    x = x.view(-1, (self.n_users + self.n_items ) * self.k)

    for i in range(len(self.fc_layers)):
      x = self.fc_layers[i](x)
      x = nn.ReLU()(x)
      x = self.dropout(x)

    x = self.output(x)
    x = self.output_f(x) * self.config['rating_range'] + self.config['lowest_rating']
    return x
