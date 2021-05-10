from preprocess import unmap_ids, map_ids, map_id, unmap_id
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


def recommend(user_id, P, Q, top_n):
  '''Returns a data frame consists of two columns, the first column
  involves the ids of top n recommendations (items or movies), given the user id

  Args:
    user_id (number): The user id
    P (np.ndarray): The First matrix P in factorization equation R = P @ Q.T
    Q (np.ndarray): The Second matrix Q in factorization equation R = P @ Q.T
    top_n (number): Number of retrieved elements

  Returns:
    pd.DataFrame: a data frame consists of two columns, the first column
  involves the ids of top n recommendations (items or movies), given the user id

  '''

  # Mapping
  user_id = map_id(user_id, user=True)

  # If user not found in mapping dictionary, we cannot recommend
  if user_id is None:
      print("Error: No users like this")
      return

  # Do recommendation
  ids, ratings= do_recommendation(user_id, P, Q, top_n)

  # Remaps the ids of movies to its original index
  result = unmap_ids(pd.Series(ids), users = False)

  result = pd.DataFrame({'Item':result.values, 'Rating':ratings})

  return result


if __name__=='__main__':

    # Return recommendations for user number 6321
    user_id = 612321
    top_n = 10

    print(f"Recommendations for User : {user_id}")

    # Loading the matrices P and Q
    MODELS_PATH = "models"

    P = np.load(path_join(MODELS_PATH, "P_ARRAY_CF.npy"))
    Q = np.load(path_join(MODELS_PATH, "Q_ARRAY_CF.npy"))

    # Call recommend utility
    df = recommend(user_id, P, Q, top_n)

    print(df)
