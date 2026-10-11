# Neural network from scratch

### perceptrons

what are they ?

lets take w1,w2,w3 weights w.r.t x1,x2,x3 inputs , weights are real number expressing importance of the respective inputs to the output.
the output is determinded by whether the weigted sum $\sum_j w_j x_j$ is less than or greater than the threshold value , threshold is real number

$$
\text{output} =
\begin{cases}
0 & \text{if } \sum_j w_j x_j \le \text{threshold} \\
1 & \text{if } \sum_j w_j x_j > \text{threshold}
\end{cases}
$$

bias = -threshold , bias is the value that is used to get to the threashold  by adding it or used to activate the perceptron.

$$
\text{output} =
\begin{cases}
0 & \text{if } w \cdot x + b \le 0 \\
1 & \text{if } w \cdot x + b > 0
\end{cases}
$$

### sigmoid neurons

Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output

the sigmoid neuron has weights for each input, $w_1, w_2, \ldots$, and an overall bias, $b$. But the output is not 0 or 1. Instead, it's $\sigma(w \cdot x + b)$, where $\sigma$ is called the _sigmoid function_
They can have as output any real number between 0 and 1, so values such as 0.173… and 0.689… are legitimate outputs

### neural network

input layer -> hidden layer -> output layer

input neurons. -> not io neurons -> output neurons

feedforward neural network : neural networks where the output from one layer is used as input to the next layer. Such networks are called _feedforward_ neural networks. This means there are no loops in the network - information is always fed forward, never fed back.

### gradient and gradient descent

first lets talk about cost function or loss

$$
C(w,b) \equiv \frac{1}{2n} \sum_x \| y(x) - a \|^2.
$$

cost is the difference from the output which we want to achieve , example if we want output neuron to be 1 but it is 0 then cost is 1

to minimise the cost , gradient descent is used.

- **Gradient** $\nabla C$ points in the direction where the cost **increases fastest**.
- **Negative gradient** $-\nabla C$ points in the direction where the cost **decreases fastest**.
- **Gradient descent** is the process of repeatedly moving in that negative-gradient direction.

Lets take an example of ball and slope
lets put the ball at minimum distance from the end of slope to make sure that it will definetly roll down to the end of slope .

lets represent weight and biases as v

$$
\nabla C \equiv \left( \frac{\partial C}{\partial v_1}, \frac{\partial C}{\partial v_2} \right)^T
$$

$$
\Delta C \approx \nabla C \cdot \Delta v
$$

$$
\Delta v = -\eta \nabla C
$$

$\eta$ is a small positive parameter (known as the _learning rate_).

to start this example lets put the ball at top , it doesnt go down to slope ? lets move it little bit further near the downward slope , do this until it is sure that ball will go downwards ,

$$
v \rightarrow v' = v - \eta \nabla C
$$

one thing to note , we dont want high learning rate since then ball can bounce the slope or not even slow learning rate else it will crawl

#### SGD , Stochastic gradient descent

instead of using entire dataset to calculate gradient , use small random group of examples called mini batch

for each input network has its own cost  and total cost is the avg of all those each cost,
to compute the gradient $\nabla C$ we need to compute the gradients $\nabla C_x$ separately for each training input $x$, and then average them,

$$
\nabla C = \frac{1}{n} \sum_x \nabla C_x
$$

now instead of doing this , we can randmoly pick training examples for small number m from $X_1$ to $X_m$ and we calulate avg gradient of that instead of full batch, and its approxiamately equal to full gradient

### code part

lets initialise neural network class with base Method

in the method we initiliase
sizes list
no of layers which is length of sizes list

The biases and weights for the network are initialized randomly, using a Gaussian distribution with mean 0, and variance 1.  Note that the first layer is assumed to be an input layer, and by convention wewon't set any biases for those neurons, since biases are only ever used in computing the outputs from later layers

```python
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
```

for given an input 'a' for the network returns corresponding output

$$
a' = \sigma(wa + b)
$$

```python
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
```

as discussed previoulsy we take training data here as list of tuples(x,y) , epochs, mini_batch , size , eta and test_data

test_data is used to evaluate the network after each epoch to check progress

in each epoch first it randomly shuffles the data and then prtition into mini_bacthes of size provided
then for each mini batch we apply single step of gradient descent `update_mini_batch(self, mini_batch, eta)` which updates the network weights and biases according to single iteration of gradient descent using just training data mini_batch

```python
   def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print ("Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete".format(j))
```

we use this under SGD function , it takes mini_batch(x,y) and eta

- `mini_batch`: list of tuples (x, y)
  - `x`: input
  - `y`: correct answer for that input
- `eta`: learning rate

we take `nabla_b` and `nabla_w`: lists of zero arrays with the same shapes as the biases and weights. and then we pass x and y for backpropagation , bapgrapopagation returns the gradient and then using that we update weight and bias.

```python
    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

```

other important code which we discuss later

```python
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

```

### backpropagation 


### resources
https://builtin.com/data-science/gradient-descent
http://neuralnetworksanddeeplearning.com/
https://knowledge.kevinjosethomas.com/Artificial-Intelligence/Gradient-Descent