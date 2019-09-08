# cuda-for-deep-learning
Transparent CUDNN / CUBLAS usage for the deep learning training using MNIST dataset.

# How to use

```bash
$ git clone https://github.com/haanjack/cudnn-mnist-training
$ cd cudnn-mnist-training
$ bash download-mnist-dataset.sh
$ make
$ ./train
```

# Expected output
```bash
== MNIST training with CUDNN ==
[TRAIN]
loading ./dataset/train-images-idx3-ubyte
loaded 60000 items..
.. model Configuration ..
CUDA: conv1
CUDA: pool
CUDA: conv2
CUDA: pool
CUDA: dense1
CUDA: relu
CUDA: dense2
CUDA: softmax
.. initialized conv1 layer ..
.. initialized conv2 layer ..
.. initialized dense1 layer ..
.. initialized dense2 layer ..
step:  200, loss: 0.561, accuracy: 75.762%
step:  400, loss: 2.754, accuracy: 96.574%
step:  600, loss: 0.157, accuracy: 97.004%
step:  800, loss: 0.005, accuracy: 97.006%
step: 1000, loss: 0.178, accuracy: 97.016%
step: 1200, loss: 0.014, accuracy: 96.998%
step: 1400, loss: 0.854, accuracy: 96.998%
step: 1600, loss: 0.165, accuracy: 96.984%
step: 1800, loss: 0.051, accuracy: 97.006%
step: 2000, loss: 0.284, accuracy: 97.025%
step: 2200, loss: 0.002, accuracy: 96.996%
step: 2400, loss: 0.013, accuracy: 96.990%
[INFERENCE]
loading ./dataset/t10k-images-idx3-ubyte
loaded 10000 items..
loss: 3.165, accuracy: 85.500%
Done.
```

# Features
* Parameter saving and loading
* Network modification
* Learning rate modificiation
* Dataset shuffling
* Testing
* Add more layers

All these features requires re-compilation
