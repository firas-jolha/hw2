from nca import NCA
from os.path import join as path_join
from read import read_data
from preprocess import map_ids, map_id
import pandas as pd
from scipy import sparse
from torch import nn
import torch.nn.functional as f
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def test_model(config):

    user_ids, movie_ids, ratings = read_data(training = False)

    user_ids = map_ids(user_ids, users = True)
    movie_ids = map_ids(movie_ids, users = False)

    users = torch.Tensor(user_ids).int()
    movies = torch.Tensor(movie_ids).int()
    ratings = torch.Tensor(ratings)

    MODELS_PATH = "models"

    all_users = np.load(path_join(MODELS_PATH, "all_users_indices.npy"))
    all_movies = np.load(path_join(MODELS_PATH, "all_movies_indices.npy"))

    # users = torch.nn.functional.one_hot(users.long(), len(all_users))
    # movies = torch.nn.functional.one_hot(movies.long(), len(all_movies))



    config['n_users'] = len(all_users)
    config['n_items'] = len(all_movies)
    config['layers'][0] = (config['n_users'] + config['n_items'])*config['k']

    # print(config)

    MODELS_PATH = "models"

    model = NCA(config)
    model.load_state_dict(torch.load(path_join(MODELS_PATH, "acf.pth"), 'cpu'))
    model.eval()

    batch_size = 200

    critertion = torch.nn.MSELoss()

    data_loader = DataLoader(TensorDataset(users, movies, ratings), batch_size = batch_size)

    losses = []
    for batch_users, batch_movies, batch_ratings in data_loader:

        batch_users = torch.nn.functional.one_hot(batch_users.long(), len(all_users))
        batch_movies = torch.nn.functional.one_hot(batch_movies.long(), len(all_movies))

        users = batch_users.int()
        movies = batch_movies.int()
        ratings = batch_ratings

        output = model(users, movies)[:,0]

        loss = critertion(output, ratings)
        losses.append(loss.item())

    print(f"Loss for test data is {np.mean(losses)}")



if __name__=="__main__":
    k = 7
    config = {
    'k': k, # Latent Space Dimension
    'layers':[-1, 64, 16, 8],  # sizes of fully connected layers
    'rating_range': 4,  # Range of rating (5 - 1 = 4)
    'lowest_rating':1 # The lowest rating (1)
    }
