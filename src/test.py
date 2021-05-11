from train import calculate_loss
import numpy as np
from scipy import sparse
import configs


def test_loss():
    """Computes the loss for test data and returns it.

    Returns
    -------
    float
		The loss of test data in data folder.
    """

    # Reading the best values for matrices P and Q from models folder
    P = np.load(configs.P_ARRAY_PATH)
    Q = np.load(configs.Q_ARRAY_PATH)

    # Reading the rating matrix for test data
    R2 = sparse.load_npz(configs.R_TEST_MATRIX_PATH)

    # Calculating the loss for test data with no Regularization
    loss = calculate_loss(P, Q, R2, lam=0)

    return loss


if __name__=="__main__":
	loss = test_loss()
	print(f"Loss for test dataset is {loss}")
