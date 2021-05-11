# Files and Models Description
This folder holds all the files which are saved during executing the scripts.

## acf.pth
A PyTorch saved file which holds the trained neural network model.

## all_movies_indices.npy
An np.ndarray holds the mapping dictionary of the movie ids.

## all_users_indices.npy
An np.ndarray holds the mapping dictionary of the user ids.

## configs.pkl
A dictionary for keeping the training configurations of the NCF model which could be used for testing purposes.

## P_ARRAY_CF.npy
An ndarray for keeping the values of the first matrix in matrix factorization such that ```R = P @ Q.T```.

## Q_ARRAY_CF.npy
An ndarray for keeping the values of the second matrix in matrix factorization such that ```R = P @ Q.T```.

## R.npz
A sparse matrix for keeping the ratings of the training data.

## R2.npz
A sparse matrix for keeping the ratings of the test data.
