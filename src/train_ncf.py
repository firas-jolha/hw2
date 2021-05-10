import torch
from torch.utils.data import TensorDataset, DataLoader
from read import read_data
from os.path import join as path_join
import numpy as np
from preprocess import map_ids
from nca import NCA

# def do_one_hot(data, n_classes):
#   batch_size = data.shape[0]
#   res = torch.zeros(batch_size, n_classes, dtype = torch.int)
#   data = data.long()
#   x = np.array(range(res.shape[0]))
#   y = np.array(data)
#   res[x, y] = 1
#   # for i in range():
#   #   res[i, data[i]] = 1
#   return res


def train(config):
    """Trains the neural network with provided configurations.

    Parameters
    ----------
    config : dictionary
        Configurations of training procedure.

    Returns
    -------
    None

    """

    # Latent Space Dimension
    k = config['k']

    # Read training data
    user_ids, movie_ids, ratings = read_data(training = True)

    user_ids = map_ids(user_ids, users = True)
    movie_ids = map_ids(movie_ids, users = False)

    # Input Data
    users = torch.Tensor(user_ids).int()
    movies = torch.Tensor(movie_ids).int()
    ratings = torch.Tensor(ratings)

    config['n_users'] = np.unique(user_ids).size
    config['n_items'] = np.unique(movie_ids).size
    config['layers'][0] = (config['n_users']+config['n_items']) * config['k']

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

        # Do one-hot encoding
        batch_users = torch.nn.functional.one_hot(batch_users.long(), config['n_users'])
        batch_movies = torch.nn.functional.one_hot(batch_movies.long(), config['n_items'])


        users = batch_users.int().to(device)
        movies = batch_movies.int().to(device)
        ratings = batch_ratings.to(device)

        optimizer.zero_grad()

        output = model(users, movies)[:, 0]

        loss = critertion(output, ratings)

        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())
        print(loss.item())

      avg_epoch_loss = np.mean(epoch_loss)
      losses.append(avg_epoch_loss)
      print(f"epoch {epoch}, loss = {avg_epoch_loss}")



    MODELS_PATH = "models"

    # Save the trained model
    path = path_join(MODELS_PATH, "acf_new.pth")
    torch.save(model.state_dict(), path)

if __name__=="__main__":
    k = 7

    config = {
       'k': k, # Latent Space Dimension
       'layers':[-1, 64, 16, 8],  # sizes of fully connected layers (first fc layer is -1 because it will be set inside training)
       'rating_range': 4,  # Range of rating (5 - 1 = 4)
       'lowest_rating':1, # The lowest rating (1)
       'lr' : 0.001,
       'batch_size': 100,
       'epochs': 10,
       'critertion': torch.nn.MSELoss()
    }

    train(config)
