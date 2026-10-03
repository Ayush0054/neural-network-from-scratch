import numpy as np

from first import NeuralNetwork
# a= np.array([1,2,3,4,5])
# print(a.shape)

# b= np.array([[1,2,3,4,5],[6,7,8,9,10]])
# print(b.shape)

# c =np.random.randn(30, 1)
# print(c)

# W = np.random.randn(30, 784)
# a = np.random.randn(784, 1) 
# b = np.dot(W, a)
# print(b.shape)

net = NeuralNetwork([784, 30, 10])
print(net.sizes)
print("--------------------------------")   
print(net.biases)
print("--------------------------------")
print(net.weights)
print("--------------------------------")
print(net.biases[0].shape)
print("--------------------------------")
print( np.zeros(net.biases[0].shape  , dtype=int))
print("--------------------------------")
print( [np.zeros(b.shape , dtype=int) for b in net.biases] ,
      "now weights",
[np.zeros(w.shape , dtype=int) for w in net.weights])
# print(net.weights[0].shape)
# print("--------------------------------")
# print(net.weights[1].shape)
# print(net.feedforward(np.array([[0.5],[0.8]])))