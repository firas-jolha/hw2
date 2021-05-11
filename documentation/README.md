# Code Documentation

This repository includes implementations of two collaborative filtering methods. Matrix factorization trained using Alternating Least Squares (ALS) and Neural Collaborative Filtering (NCF).

The repository mainly consists of 8 scripts for training and testing the models as well as other utilities.

## read.py 
- includes ```read_data``` function for reading the dataset, this function accepts ```training``` parameter for specifying the kind of data (training or test data). 

## preprocess.py
- has 4 functions for mapping and remapping ids of users and movies. 
- ```map_ids``` function resets the ids to the basic index [0-n] where n is the number of unique ids in the list. 
This function accepts two parameters, ```series``` a series of ids to reset and ```users``` boolean variable to specify users and movies.

- ```unmap_ids``` function returns the ids back to their original index, to specify the ids of users or movies, a boolean parameter ```users``` is used. 

- ```map_id``` function accepts a single ID of a user or a movie and returns the new mapped index. The parameter ```user``` is True for user ID and Flase for movie ID. 

- ```unmap_id``` function does the reverse job of ```map_id```. 

- ```create_mapping``` function builds the mapping dictionary for the given list of ids, and returns the mapping array which could be persisted for later use. 

- ```preprocess``` function which reads the training and test datasets, then preprocess them and returns the sparse rating matrices R for training data and R2 for test data. 

## train.py

- has the main functions for training the basic collaborative filtering method.

- ```train``` function which accepts ```config``` parameter for passing the settings of training procedure. It does the main job of preparing the dataset then do training for a number of epochs. The resulted arrays P and Q are saved for later use.

- ```update_P``` function which is used for updating the values of the matrix P for ALS algorithm.

- ```update_Q``` function which is used for updating the values of the matrix Q for ALS algorithm.

- ```claculate_loss``` function for computing the loss of the current P and Q matrices. 



## test.py

- has only one function called ```test_loss```. The objective of this file is to apply the learned P and Q of training on the test data. 

- ```test_loss``` function loads the matrices P and Q which are saved in training stage and calculates the loss on the test data. 

## train_ncf.py

- used for training the neural collaborative filtering. 


## test_ncf.py
- used for testing the neural network model. 

## recommend.py
- used for making recommendations for users given their ids. It has two functions ``` recommend``` and helper function ```do_recommendation```. 

- ```recommend``` function takes the user ID, the matrices P and Q as well as the number of recommendations to be retrieved ```top_n```. Firstly, it maps or resets the ID of the given user, then predict the ratings using P and Q matrices. After that only top N recommendations or movies are retrieved. But these ids need to be remapped to their original index. This function returns a Data Frame of two columns, the first column holds the recommended item or movie ids and the second column has the ratings of them. 

- ```do_recommendation``` function is the backbone of ```recommend``` function. It is used by ```recommend``` function for retrieving the ratings for the given user. 

## nca.py

- includes ```NCA``` class which defines the neural collaborative filtering models.
