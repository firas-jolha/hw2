import pandas as pd
from os.path import join as path_join
import numpy as np
from scipy import sparse

def map_userid(id):

    pass

def unmap_userid(id):

    pass


def map_ids(series):
    """Resets ids of both users and items to the range [0-n].

    Parameters
    ----------
    series : pd.Series
        The series of ids to be reset.

    Returns
    -------
    pd.Series
        A series of the same kind after resetting the ids.

    """

    # Taking the unique values of the input
    uq = series.unique()

    # remapping the values
    return series.map(pd.Series(range(0, uq.size), index = uq))

def unmap_ids(series, original_series):
    """Returns back the original ids of series being converted by map_ids function given the original series.

    Parameters
    ----------
    series : pd.Series
        The series of ids to be returned back.
    original_series : pd.Series
        The series of ids from the original data frame.

    Returns
    -------
    pd.Series
        A series contains the original ids.

    """
    # Taking the unique values
    uq = original_series.unique()

    # Remapping
    return series.map(pd.Series(uq, index = range(0, uq.size)))


def preprocess():
    """Reads the dataset from data folder then preprocess it and returns the results of both training and test dataset.

    Returns
    -------
    R : scipy.sparse.coo_matrix
        Sparse matrix for ratings of training data
    R2 : scipy.sparse.coo_matrix
        Sparse matrix for ratings of test data

    """
    # user_ids : pd.Series
    #     User ids after resetting.
    # movie_ids : pd.Series
    #     Movie ids after resetting.
    # ratings : pd.series
    #     Ratings of users to movies for training purposes.
    # n_users : np.int
    #     number of unique users in the training dataset.
    # n_movies : np.int
    #     number of unique movies in the training dataset.


    # Set Data Path
    DATA_PATH = "data"

    # Read Training and Test Data
    train_df = pd.read_csv(path_join(DATA_PATH, "train.csv"))
    test_df = pd.read_csv(path_join(DATA_PATH, "test.csv"))

    # Data Exploration and Preprocessing
    user_ids = train_df['userId']
    movie_ids = train_df['movieId']
    ratings = train_df['rating']


    # Resetting the ids of training data to [0-n]
    user_ids = map_ids(user_ids)
    movie_ids = map_ids(movie_ids)



    # Resetting the ids of test data
    test_user_ids = map_ids(test_df['userId'])
    test_movie_ids = map_ids(test_df['movieId'])
    test_ratings = test_df['rating']

    # Statistics of training data
    # Number of users and movies can be extracted from the array of mapped ids
    n_users = np.max(user_ids) + 1
    n_movies = np.max(movie_ids) + 1

    # Statistics of test data
    test_n_users = np.max(test_user_ids) + 1
    test_n_movies = np.max(test_movie_ids) + 1

    # Returning the indices back can be done using unmap_ids function
    # Example
    # unmap_ids(movie_ids, train_df['movieId'])
    # unmap_ids(user_ids, train_df['userId'])

    # Define the training rating matrix as sparse matrix

    # Sparse rating matrix from training data
    R = sparse.coo_matrix(
        (ratings, (user_ids, movie_ids)),
        shape=(n_users, n_movies),
        dtype=np.float
     )

    MODELS_PATH = 'models'

    # Save the rating matrix for training data
    sparse.save_npz(path_join(MODELS_PATH,'R.npz'), R)

    # Sparse rating matrix from test data
    R2 = sparse.coo_matrix(
        (test_ratings, (test_user_ids, test_movie_ids)),
        shape=(test_n_users, test_n_movies),
        dtype=np.float
    )

    # Save the rating matrix for test data
    sparse.save_npz(path_join(MODELS_PATH,'R2.npz'), R2)

    # return user_ids, movie_ids, ratings, n_users, n_movies, R, R2
    return R, R2


if __name__=='__main__':
    R, R2 = preprocess()
    print(f'Rating matrix for training : {R.shape}')
    print(f'# Users in train set : {R.shape[0]}')
    print(f'# Movies in train set : {R.shape[1]}')
    print(f'Rating matrix for testing : {R2.shape}')
    print(f'# Users in test set : {R2.shape[0]}')
    print(f'# Movies in test set : {R2.shape[1]}')
