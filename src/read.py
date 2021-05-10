import pandas as pd
from os.path import join as path_join

def read_data(training = True):

    # Set Data Path
    DATA_PATH = "data"

    if training:
        # Read Training data
        df = pd.read_csv(path_join(DATA_PATH, "train.csv"))
    else:
        # Read Test data
        df = pd.read_csv(path_join(DATA_PATH, "test.csv"))

    user_ids = df['userId']
    item_ids = df['movieId']
    ratings = df['rating']

    return user_ids, item_ids, ratings
