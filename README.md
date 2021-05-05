# HW2
Advanced Machine Learning Course - Homework 2

![Docker Image Test](https://github.com/firas-jolha/hw2//actions/workflows/docker-image.yml/badge.svg)

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
- For testing the models run
```bash
python3 ./src/test.py
```
