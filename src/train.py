# Train Script
from preprocess import preprocess
import pandas as pd
from scipy import sparse
from os.path import join as path_join
import numpy as np


def update_P(P, Q, R, alpha = 0.001, lam = 0.001):
    """Updates the values of matrix P.

        Parameters
        ----------
        P: np.ndarray
        	the first matrix in factorization
        Q: np.ndarray
        	the second matrix in factorization
        R: scipy.sparse.coo_matrix
        	sparse ratings matrix
        alpha: np.float
        	Learning rate
        lam: np.float
        	Regularization rate

        Returns
        -------
        np.ndarray
        	P itself after adjusting the values

    """
    assert P.shape[1] == Q.shape[1], "P and Q should have proper dimensions for matrix multiplication"

    M = np.zeros(R.shape)

    x1, x2 = R.nonzero()

    # M[x1, x2] = 1

    P_tau = P[x1, :]
    Q_tau = Q[x2, :]

    # Inner Product
    prod = np.sum((P_tau * Q_tau), axis = 1)

    R_hat = sparse.coo_matrix((prod, (x1, x2)), shape = R.shape)

    res = R_hat - R

    gradient = alpha * ((res @ Q) - lam * P)

    P += - gradient

    return P


def update_Q(P, Q, R, alpha = 0.001, lam = 0.001):
    """Updates the values of matrix Q.

    Parameters
    ----------
    P: np.ndarray
		the first matrix in factorization
	Q: np.ndarray
		the second matrix in factorization
	R: scipy.sparse.coo_matrix
		sparse ratings matrix
	alpha: np.float
		Learning rate
	lam: np.float
		Regularization rate

    Returns
    -------
    np.ndarray
		Q itself after adjusting the values
    """

    assert P.shape[1] == Q.shape[1], "P and Q should have proper dimensions for matrix multiplication"

    M = np.zeros(R.shape)
    x1, x2 = R.nonzero()

    # # Mask
    # M[x1, x2] = 1


    P_tau = P[x1, :]
    Q_tau = Q[x2, :]

    # Inner Product
    prod = np.sum((P_tau * Q_tau), axis = 1)

    R_hat = sparse.coo_matrix((prod, (x1, x2)), shape = R.shape)

    res = R_hat - R

    gradient = alpha * ((res.T @ P) - lam * Q )

    Q += - gradient


    return Q

def calculate_loss(P, Q, R, lam):
    """Calculates the loss for ALS algorithm.

    Parameters
    ----------
    P: np.ndarray
		the first matrix in factorization
	Q: np.ndarray
		the second matrix in factorization
	R: scipy.sparse.coo_matrix
		sparse ratings matrix
	lam: np.float
		Regularization rate

    Returns
    -------
    res : np.float
        The current loss of training.

    """

    R_prod = (P @ Q.T)

    x1, x2 = R.nonzero()
    R_hat = R_prod[x1, x2]
    R_tau = R.data

    mu = np.mean(R_hat) # Overall average rating
    b_u = np.mean(R_prod, axis = 1)
    b_i = np.mean(R_prod, axis = 0)
    b_u = b_u[x1] - mu
    b_i = b_i[x2] - mu

    res = (R_hat + mu + b_u + b_i) - R_tau

    res = np.square(res)

    res = np.mean(res)

    res += lam *(np.linalg.norm(P) + np.linalg.norm(Q)) # Regularization Term

    return res


def train(config):

	# Do the Preprocessing
	R, R2 = preprocess()

	# number of users in trainset
	n_users = R.shape[0]

	# number of movies in trainset
	n_movies = R.shape[1]

	# Set the dimension of latent space
	k = config['k']

	# Initialize the matrices P and Q randomly
	P = np.random.random(size = (n_users, k))
	Q = np.random.random(size = (n_movies, k))

	# Learning Rate
	lr = config['lr']

	# Regularization rate
	lam = config['lambda']

	# Variables to keep the best P and Q according to the lowest test loss
	best_P = P
	best_Q = Q
	last_loss = None
	epochs = config['epochs']

	for iter in range(epochs):

	  # Update Steps for P then Q by following ALS algorithm
	  P = update_P(P, Q, R, alpha=lr, lam=lam)
	  Q = update_Q(P, Q, R, alpha=lr, lam=lam)

	  # Regularization rate lam = 0 for calculating the loss of test data R2
	  # Regularization term is not included in test error
	  test_loss = calculate_loss(P, Q, R2, lam=0)

	  # Calculating the loss of training data R
	  loss = calculate_loss(P, Q, R, lam=lam)

	  print(f" epoch {iter}, training loss {loss}, test loss {test_loss} ")

	  # A control block for keeping the best P and Q for the lowest test loss during the training
	  if last_loss:
	    if last_loss > test_loss:
	      best_P = P.copy()
	      best_Q = Q.copy()
	      last_loss = test_loss
	    else:
	      pass

	  else:
	    last_loss = test_loss

	print(f"Lowest loss reached for test data is {last_loss}")

	MODELS_PATH = "models"

	# Persisting the best matrices P and Q
	with open(path_join(MODELS_PATH, "P_ARRAY_CF_new.npy"), "wb") as f:
	  np.save(f, best_P)
	with open(path_join(MODELS_PATH, "Q_ARRAY_CF_new.npy"), "wb") as f:
	  np.save(f, best_Q)




if __name__ == "__main__":
	config = {
	'k': 7,
	'lr': 1e-6,
	'lambda': 1e-12,
	'epochs': 30,
	}

	# Do Training
	train(config)
	print("Training is Finished!")
