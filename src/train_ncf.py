import torch
from torch.utils.data import TensorDataset, DataLoader
from read import read_data
import numpy as np
from preprocess import map_ids
from nca import NCA
import pickle
import configs


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

    # The input size of first fc layer is different if we do one hot encoding for the input
    if config['one_hot_encoding']:
        config['layers'][0] = (config['n_users'] + config['n_items']) * config['k']
    else: # 2 by k - because we conactenate the users and items, k is the output size of embedding layers
        config['layers'][0] = 2 * config['k']

    print("Configurations")
    print(config)

    # Save the configs as a dictionary
    with open(configs.CONFIGS_PATH, "wb") as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)


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

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters())

    # Create a data loader from training data
    data_loader = DataLoader(TensorDataset(users, movies, ratings), batch_size = batch_size)

    # Accumulatas the loss across epochs
    losses = []

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    print("-"*50)

    # Iterate over epochs
    for epoch in epochs:
      epoch_loss = []


      # Iterate over batches
      for batch_users, batch_movies, batch_ratings in data_loader:

        # Do one-hot encoding
        if config['one_hot_encoding']:
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

      avg_epoch_loss = np.mean(epoch_loss)
      losses.append(avg_epoch_loss)
      print(f"epoch {epoch}, loss = {avg_epoch_loss}")

    # Save the trained model
    # Save different models to different files which is based whether it includes one-hot encoding of features or not
    if config['one_hot_encoding']:
        torch.save(model.state_dict(), configs.NCF_MODEL_ONE_HOT_PATH)
    else:
        torch.save(model.state_dict(), configs.NCF_MODEL_PATH)



if __name__=="__main__":

    # Training Settings
    config = {
       'k': 7, # Latent Space Dimension
       'layers':[-1, 64, 16, 8, 4],  # sizes of fully connected layers (first fc layer is -1 because it will be set inside training)
       'rating_range': 4,  # Range of rating (5 - 1 = 4)
       'lowest_rating':1, # The lowest rating (1)
       'lr' : 0.001, # Learning Rate
       'batch_size': 1000, # Batch Size
       'epochs': 10, # Number of epochs
       'critertion': torch.nn.MSELoss(), # Loss function
       'one_hot_encoding': False # One hot encoding of features
    }

    # Do Training
    train(config)
