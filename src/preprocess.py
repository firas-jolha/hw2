import pandas as pd
import numpy as np
from scipy import sparse
import configs

def map_id(id, user = True):
    """Maps the id of user or movie to basic indexing [0-n].

    Parameters
    ----------
    id : int
        user or movie id.
    user : boolean
        True for mapping user id and False for mapping movie id.

    Returns
    -------
    int
        the id after mapping.

    """

    # Reading the mapping array from models folder

    uq = None
    if user:
        uq = np.load(configs.USER_IDS_PATH)
    else:
        uq = np.load(configs.MOVIE_IDS_PATH)


    # Returning the new id
    if id in uq:
        return np.where(uq==id)[0][0]
    else:
        return None


def unmap_id(id, user=True):
    """Returns the original ids of user or movie.

    Parameters
    ----------
    id : int
        user or movie id.
    user : boolean
        True for remapping user id and False for remapping movie id.

    Returns
    -------
    int
        the id after remapping.

    """

    # Reading the mapping array from models folder

    uq = None
    if user:
        uq = np.load(configs.USER_IDS_PATH)
    else:
        uq = np.load(configs.MOVIE_IDS_PATH)


    # Returning the new id
    if id < uq.size:
        return uq[id]
    else:
        return None


def map_ids(series, users = True):
    """Resets ids of both users and items to the range [0-n].

    Parameters
    ----------
    series : pd.Series
        The series of ids to be reset.
    users : boolean
        True for mapping user ids and False for mapping movie ids.

    Returns
    -------
    pd.Series
        A series of the same kind after resetting the ids.

    """

    # Taking the unique values of the input
    # uq = series.unique()

    uq = None
    if users:
        uq = np.load(configs.USER_IDS_PATH)
    else:
        uq = np.load(configs.MOVIE_IDS_PATH)

    # remapping the values
    return series.map(pd.Series(range(0, uq.size), index = uq))

def unmap_ids(series, users = True):
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


    uq = None
    if users:
        uq = np.load(configs.USER_IDS_PATH)
    else:
        uq = np.load(configs.MOVIE_IDS_PATH)
    # uq = original_series.unique()

    # Remapping
    return series.map(pd.Series(uq, index = range(0, uq.size)))

def create_mapping(series):
    """Creates the mapping for given series.

    Parameters
    ----------
    series : pd.Series
        The original series to be mapped.

    Returns
    -------
    uq : pd.Series
        The unqiue values in the series after sorting.

    """
    uq = np.sort(series.unique())
    return uq

def preprocess():
    """Reads the dataset from data folder then preprocess it and returns the results of both training and test dataset.

    Returns
    -------
    R : scipy.sparse.coo_matrix
        Sparse matrix for ratings of training data
    R2 : scipy.sparse.coo_matrix
        Sparse matrix for ratings of test data

    """

    # Read Training and Test Data
    train_df = pd.read_csv(configs.TRAIN_DATA_PATH)
    test_df = pd.read_csv(configs.TEST_DATA_PATH)

    # Data Exploration and Preprocessing
    user_ids = train_df['userId']
    movie_ids = train_df['movieId']
    ratings = train_df['rating']

    # Create the mapping and save it for later usage in testing
    all_users = create_mapping(pd.concat([train_df['userId'], test_df['userId']], axis = 0))
    all_movies = create_mapping(pd.concat([train_df['movieId'], test_df['movieId']], axis = 0))


    # Save the mapping arrays for users and movies
    with open(configs.USER_IDS_PATH, "wb") as f:
        np.save(f, all_users)

    with open(configs.MOVIE_IDS_PATH, "wb") as f:
        np.save(f, all_movies)


    # Resetting the ids of training data to [0-n]
    user_ids = map_ids(user_ids, users = True)
    movie_ids = map_ids(movie_ids, users = False)


    # Resetting the ids of test data
    test_user_ids = map_ids(test_df['userId'], users = True)
    test_movie_ids = map_ids(test_df['movieId'], users = False)
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
    # unmap_ids(movie_ids, users = False)
    # unmap_ids(user_ids, users = True)

    # Define the training rating matrix as sparse matrix

    # Sparse rating matrix from training data
    R = sparse.coo_matrix(
        (ratings, (user_ids, movie_ids)),
        shape=(n_users, n_movies),
        dtype=np.float
     )



    # Save the rating matrix for training data
    sparse.save_npz(configs.R_TRAIN_MATRIX_PATH, R)

    # Sparse rating matrix from test data
    R2 = sparse.coo_matrix(
        (test_ratings, (test_user_ids, test_movie_ids)),
        shape=(test_n_users, test_n_movies),
        dtype=np.float
    )

    # Save the rating matrix for test data
    sparse.save_npz(configs.R_TEST_MATRIX_PATH, R2)

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
    # print(map_id(118696, user=False))
