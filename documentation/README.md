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

- ```create_mapping``` function builds the mapping dictionary and persists it in an array for later use. 

- ```preprocess``` function which reads the training and test datasets, then preprocess them and returns the sparse rating matrices R for training data and R2 for test data. 

## train.py

## test.py

## train_ncf.py

## test_ncf.py

## recommend.py

## nca.py
