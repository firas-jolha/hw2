# User-Based Collaborative Filtering
Advanced Machine Learning Course - Homework 2

![Docker Image Test](https://github.com/firas-jolha/hw2//actions/workflows/docker-image.yml/badge.svg)

---
# Description
This program is an implementation collaborative filtering technique in recommender systems. The idea is to build a full ratings matrix for users and movies. This repo includes implementations of two methods for imputing the lost ratings of users to movies.

## 1. Matrix Factorization using Alternating Least Squares
 
 This method tries to factorize the ratings matrix into two matrices P and Q. This mehtods learns the embeddings using ALS algorithm by training for some iterations by minimizing the objective function.

## 2. Neural Collaborative Filtering
This method uses an artificial neural network to learn the latent space of the ratings then predicting them.

---

# Installation

You have two options to run this program.

## Running in Docker

To run this program in a docker container, run the following commands:

1. Build the image
```bash
docker build -t hw2 .
```

2. Run the container 
```bash
docker run hw2
```

## Running Locally

To run this program locally, you need to do as follows:

1. Install the requirements
```bash
pip3 install -r requirements.txt
```

2. Execute scripts
- For training the model
```bash
python3 ./src/train.py
```
- For testing the model
```bash
python3 ./src/test.py
```
