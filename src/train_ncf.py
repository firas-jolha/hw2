import pandas as pd
from scipy import sparse
from torch import nn
import torch.nn.functional as f
import torch
from torch.utils.data import TensorDataset, DataLoader
from read import read_data
from os.path import join as path_join
import numpy as np
from preprocess import map_ids


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

    for i in range(len(self.fc_layers)):
      x = self.fc_layers[i](x)
      x = nn.ReLU()(x)
      x = self.dropout(x)

    x = self.output(x)
    x = self.output_f(x) * config['rating_range'] + config['lowest_rating']
    return x


def train(config):

    # Latent Space Dimension
    k = config['k']

    # Read training data
    user_ids, movie_ids, ratings = read_data(training = True)

    user_ids = map_ids(user_ids)
    movie_ids = map_ids(movie_ids)

    # Input Data
    users = torch.Tensor(user_ids).int()
    movies = torch.Tensor(movie_ids).int()
    ratings = torch.Tensor(ratings)

    config['n_users'] = np.unique(user_ids).size
    config['n_items'] = np.unique(movie_ids).size

    print("Configurations")
    print(config)

    # Try to use GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = NCA(config).to(device)

    print("-"*50)
    print("Our Model")
    print(model)

    learning_rate = config['lr']
    critertion = config['critertion']
    batch_size = config['batch_size']
    epochs = range(config['epochs'])

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters())

    # Create a data loader from training data
    data_loader = DataLoader(TensorDataset(users, movies, ratings), batch_size = batch_size)

    # Accumulatas the loss across epochs
    losses = []

    print("-"*50)

    # Iterate over epochs
    for epoch in epochs:
      epoch_loss = []

      # Iterate over batches
      for batch_users, batch_movies, batch_ratings in data_loader:

        users = batch_users.to(device)
        movies = batch_movies.to(device)
        ratings = batch_ratings.to(device)

        optimizer.zero_grad()

        output = model(users, movies)[:, 0]

        loss = critertion(output, ratings)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

      avg_epoch_loss = np.mean(epoch_loss)
      losses.append(avg_epoch_loss)
      print(f"epoch {epoch}, loss = {avg_epoch_loss}")



    MODELS_PATH = "models"
    # Save the trained model
    path = path_join(MODELS_PATH, "acf_new.pth")
    torch.save(model.state_dict, path)

if __name__=="__main__":
    k = 7
    # sparse.load_npz()
    config = {
       'k': k, # Latent Space Dimension
       'layers':[k * 2, 64, 16, 8],  # sizes of fully connected layers
       'rating_range': 4,  # Range of rating (5 - 1 = 4)
       'lowest_rating':1, # The lowest rating (1)
       'lr' : 0.001,
       'batch_size': 100,
       'epochs': 4,
       'critertion': nn.MSELoss()
    }

    train(config)
