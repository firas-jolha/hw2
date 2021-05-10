from preprocess import unmap_ids, map_ids
import pandas as pd
from scipy import sparse
from os.path import join as path_join
import numpy as np

def do_recommendation(user, P, Q, n=5):
  '''Returns a list of top n recommendations (movies or items) given id of the user

  Args:
    user (number): The user id
    P (np.ndarray): The First matrix P in factorization equation R = P @ Q.T
    Q (np.ndarray): The Second matrix Q in factorization equation R = P @ Q.T
    n (number): Number of retrieved elements

  Returns:
    list: a list of ids of top n recommendations
    list: a list of ratings of top n recommendations

  '''

  # Calculate predicted ratings
  R_hat = P @ Q.T

  # Select the ratings of the specific user
  ratings = R_hat[user, :]

  # Returns the top n ratings
  ids = np.argsort(ratings)[-n:][::-1]

  return ids, ratings[ids]


def recommend(user_id, P, Q, top_n, original_item_ids):
  '''Returns a data frame consists of two columns, the first column
  involves the ids of top n recommendations (items or movies), given the user id

  Args:
    user_id (number): The user id
    P (np.ndarray): The First matrix P in factorization equation R = P @ Q.T
    Q (np.ndarray): The Second matrix Q in factorization equation R = P @ Q.T
    top_n (number): Number of retrieved elements
    original_item_ids: a list of original ids of the items before mappingthem to basic indexing

  Returns:
    pd.DataFrame: a data frame consists of two columns, the first column
  involves the ids of top n recommendations (items or movies), given the user id

  '''

  # Do recommendation
  ids, ratings= do_recommendation(user_id, P, Q, top_n)

  # Remaps the ids of movies to its original index
  result = unmap_ids(pd.Series(ids), original_item_ids)

  result = pd.DataFrame({'Item':result.values, 'Rating':ratings})

  return result


# Usage



if __name__=='__main__':

    # Return recommendations for user number 6321
    user_id = [1]
    top_n = 10

    print(f"Recommendations for User : {user_id[0]}")

    MODELS_PATH = "models"

    # R2 = sparse.load_npz(path_join(MODELS_PATH, "R.npz"))
    # original_user_ids, original_item_ids = R2.nonzero()


    # Set Data Path
    DATA_PATH = "data"

    # Read Training and Test Data
    test_df = pd.read_csv(path_join(DATA_PATH, "train.csv"))
    original_user_ids = test_df['userId']
    original_item_ids = test_df['movieId']

    user_id = pd.Series(user_id)
    user_id = map_ids(user_id)
    user_id = user_id[0]

    print(np.sort(original_item_ids))
    print(user_id)

    P = np.load(path_join(MODELS_PATH, "P_ARRAY_CF.npy"))
    Q = np.load(path_join(MODELS_PATH, "Q_ARRAY_CF.npy"))

    df = recommend(user_id, P, Q, top_n, original_item_ids)

    print(df)
