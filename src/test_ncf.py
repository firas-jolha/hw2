from nca import NCA
from read import read_data
from preprocess import map_ids
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import configs

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

    config = {}
    with open(configs.CONFIGS_PATH, "rb") as f:
        config = pickle.load(f)

    model = NCA(config)
    model.load_state_dict(torch.load(configs.NCF_MODEL_PATH, 'cpu'))
    model.eval()

    batch_size = 200

    critertion = config['critertion']

    data_loader = DataLoader(TensorDataset(users, movies, ratings), batch_size = batch_size)

    losses = []

    for batch_users, batch_movies, batch_ratings in data_loader:

        batch_users = torch.nn.functional.one_hot(batch_users.long(), config['n_users'])
        batch_movies = torch.nn.functional.one_hot(batch_movies.long(), config['n_items'])

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
