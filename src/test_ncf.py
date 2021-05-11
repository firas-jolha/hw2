from nca import NCA
from read import read_data
from preprocess import map_ids
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import configs

def test_model():
    """Runs the persisted neural network model on test data and returns the test loss.

    Returns
    -------
    None

    """

    user_ids, movie_ids, ratings = read_data(training = False)

    # Resetting ids of users and movies
    user_ids = map_ids(user_ids, users = True)
    movie_ids = map_ids(movie_ids, users = False)

    # Creating the tensors
    users = torch.Tensor(user_ids).int()
    movies = torch.Tensor(movie_ids).int()
    ratings = torch.Tensor(ratings)

    # Reading the training settings to be fed into the model
    config = {}
    with open(configs.CONFIGS_PATH, "rb") as f:
        config = pickle.load(f)

    model = NCA(config)

    # Different models for different Preprocessing steps
    if config['one_hot_encoding']:
        model.load_state_dict(torch.load(configs.NCF_MODEL_ONE_HOT_PATH, 'cpu'))
    else:
        model.load_state_dict(torch.load(configs.NCF_MODEL_PATH, 'cpu'))

    model.eval()

    # Batch size for test data
    batch_size = 200

    # The same critertion used for training stage
    critertion = config['critertion']

    # Creating loader for test data
    data_loader = DataLoader(TensorDataset(users, movies, ratings), batch_size = batch_size)

    losses = []

    # Iterate over batches
    # We calculate the test loss on batches due to the large dataset size
    for batch_users, batch_movies, batch_ratings in data_loader:

        # Whether we have to do one hot encoding or not
        if config['one_hot_encoding']:
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
