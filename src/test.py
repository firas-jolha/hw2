# Test Script
from train import calculate_loss
import numpy as np
from os.path import join as path_join
from scipy import sparse

def test_loss():
    """Computes the loss for test data and returns it.

    Returns
    -------
    float
		The loss of test data in data folder.
    """

    # Models path where saved data containers and models live
    MODELS_PATH = 'models'

    # Reading the best values for matrices P and Q from models folder
    P = np.load(path_join(MODELS_PATH, "P_ARRAY_CF.npy"))
    Q = np.load(path_join(MODELS_PATH, "Q_ARRAY_CF.npy"))

    # Reading the rating matrix for test data
    R2 = sparse.load_npz(path_join(MODELS_PATH, "R2.npz"))

    # Calculating the loss for test data with no Regularization
    loss = calculate_loss(P, Q, R2, lam=0)

    return loss


if __name__=="__main__":
	loss = test_loss()
	print(f"Loss for test dataset is {loss}")
