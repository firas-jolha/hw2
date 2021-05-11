import pandas as pd
import configs

def read_data(training = True):
    """Reads the data from data folder.

    Parameters
    ----------
    training : boolean
        True for reading training data and False for reading test data.

    Returns
    -------
    user_ids : pd.Series
        series of user ids.
    movie_ids : pd.Series
        series of movie ids.
    ratings : pd.Series
        series of ratings.

    """

    if training:
        # Read Training data
        df = pd.read_csv(configs.TRAIN_DATA_PATH)
    else:
        # Read Test data
        df = pd.read_csv(configs.TEST_DATA_PATH)

    user_ids = df['userId']
    item_ids = df['movieId']
    ratings = df['rating']

    return user_ids, item_ids, ratings
