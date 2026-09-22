import numpy as np
class NeuralNetwork:
   """intialising neural network class we initilaise method
    with no of layers which is length of sizes list and 
    we initialise biases and weights as matrices 
   """

   def __init__(self,sizes):
       self.num_layers = len(sizes)
       self.sizes = sizes
       self.biases = [np.random.randn(y,1) for y in sizes[1:]] 
       self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

