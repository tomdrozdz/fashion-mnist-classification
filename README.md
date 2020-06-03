# fashion-mnist-classification

## Introduction
This project aims to create a model that can classify images of clothes from the Fashion-MNIST dataset with the best possible accuracy. The dataset itself contains a training set of 60 000 examples and a test set of 10 000 examples. Every one of the examples consists of a 28x28 image and a label specifying one of the 10 possible classes. To read more about the dataset, visit the [Fashion-MNIST repository](https://github.com/zalandoresearch/fashion-mnist).


## Methods

### k-nearest neighobrs
Firstly, the k-NN algorithm (see [`knn.py`](knn.py)) was used to classify the images, with the value of k set to 5. Manhattan distance was used to measure the distance between images. Those choices were made mostly in order to later compare the results to bechmarks available [here](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/).

### network


## Results

### k-nearest neighbors
The k-NN algorithm achieved an accuracy of **86.23%**. 
According to the [benchmarks](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/), the same classifier from the sklearn library, with the same settings, achieved an accuracy of 86.0%. This small deviation can be probably attributed to minor differences in the implementations (for example, which class the algorithm chooses if both of them have the same probability) and no serious conlusions should be drawn from it.

### network

## Usage

### k-nearest neighbors
If you just want to check the k-NN results, then prepare ~10GBs of RAM, run [`knn.py`](knn.py) (you only need numpy and the data) and wait around an hour.

### network
All the necessary Python libraries are listed in the [`requirements.txt`](requirements.txt) file.

To install them, run:

	pip install -r requirements.txt

or if you are using conda:

	conda install --file requirements.txt

The data used to train and test the network is downloaded automatically into a specified path (default: "./data") by the [`data.py`](data.py) script before doing anything else.





## Links
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

[Some benchmarks](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/)
