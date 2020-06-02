# fashion-mnist-classification

## Introduction

## Methods

## Results

### k-nearest neighbors
The k-NN algoritm (see [`knn.py`](knn.py)) was tested with value of k set to 5. Manhattan distance was used to measure the distance between the images. The algorithm achieved an accuracy of 86.23%.

### network

## Usage
All the necessary Python libraries are listed in the [`requirements.txt`](requirements.txt) file.

To install them, run:

	pip install -r requirements.txt

or if you are using conda:

	conda install --file requirements.txt

The data used to train and test the network is downloaded automatically into a specified path (default: "./data") by the [`get_data.py`](get_data.py) script before using any other script.

## Links
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
[Some benchmarks](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)
