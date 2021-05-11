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
import pickle
from configs import MODELS_PATH

def test_model():
    """Runs the saved neural network model on test data and returns the test loss.

    Returns
    -------
    None

    """

    user_ids, movie_ids, ratings = read_data(training = False)

    user_ids = map_ids(user_ids, users = True)
    movie_ids = map_ids(movie_ids, users = False)

    users = torch.Tensor(user_ids).int()
    movies = torch.Tensor(movie_ids).int()
    ratings = torch.Tensor(ratings)

    all_users = np.load(path_join(MODELS_PATH, "all_users_indices.npy"))
    all_movies = np.load(path_join(MODELS_PATH, "all_movies_indices.npy"))

    config = {}
    with open(path_join(MODELS_PATH, "configs.pkl"), "rb") as f:
        config = pickle.load(f)

    # users = torch.nn.functional.one_hot(users.long(), len(all_users))
    # movies = torch.nn.functional.one_hot(movies.long(), len(all_movies))



    config['n_users'] = len(all_users)
    config['n_items'] = len(all_movies)
    config['layers'][0] = (config['n_users'] + config['n_items'])*config['k']


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

    # Test the NCF
    test_model()
