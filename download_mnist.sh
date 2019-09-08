#!/bin/sh

url_base=http://yann.lecun.com/exdb/mnist

mkdir -p dataset

wget ${url_base}/train-images-idx3-ubyte.gz
wget ${url_base}/train-labels-idx1-ubyte.gz
wget ${url_base}/t10k-images-idx3-ubyte.gz
wget ${url_base}/t10k-labels-idx1-ubyte.gz

mv *.gz dataset
gunzip dataset/*.gz