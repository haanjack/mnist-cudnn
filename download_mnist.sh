#!/bin/sh

url_base=http://yann.lecun.com/exdb/mnist
dataset_path=dataset

mkdir -p ${dataset_path}

wget -P ${dataset_path} ${url_base}/train-images-idx3-ubyte.gz
wget -P ${dataset_path} ${url_base}/train-labels-idx1-ubyte.gz
wget -P ${dataset_path} ${url_base}/t10k-images-idx3-ubyte.gz
wget -P ${dataset_path} ${url_base}/t10k-labels-idx1-ubyte.gz

gunzip dataset/*.gz
