# Files and Models Description
This folder holds all the files which are saved during executing the scripts.

## acf.pth
The trained neural network model without using one hot encoding for the input features.


## acf_oh.pth
The trained neural network model by using one hot encoding for the input features.

## all_users_indices.npy
An np.ndarray holds the mapping dictionary of the user ids.

## all_movies_indices.npy
An np.ndarray holds the mapping dictionary of the movie ids.

## configs.pkl
A dictionary for keeping the training configurations of the NCF model which could be used for testing purposes.


## configs_oh.pkl
A dictionary for keeping the training configurations of the NCF model with one hot encoding for the input features.

## P_ARRAY_CF.npy
An ndarray for keeping the values of the first matrix in matrix factorization such that ```R = P @ Q.T```.

## Q_ARRAY_CF.npy
An ndarray for keeping the values of the second matrix in matrix factorization such that ```R = P @ Q.T```.

## R.npz
A sparse matrix for keeping the ratings of the training data.

## R2.npz
A sparse matrix for keeping the ratings of the test data.
