NEural network from scratch

### perceptrons 
what are they ?   
lets take w1,w2,w3 weights w.r.t x1,x2,x3 inputs , weights are real number expressing importance of the respective inputs to the output.
the output is determinded by whether the weigted sum ∑j wjxj is less than or greater than the threshold value , threshold is real number 

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable columnalign="right center left" rowspacing="3pt" columnspacing="0 thickmathspace" displaystyle="true">
    <mlabeledtr>
      <mtd>
        <mstyle displaystyle="false" scriptlevel="0">
          <mtext>output</mtext>
        </mstyle>
      </mtd>
      <mtd>
        <mi></mi>
        <mo>=</mo>
      </mtd>
      <mtd>
        <mrow>
          <mo>{</mo>
          <mtable columnalign="left left" rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <mn>0</mn>
              </mtd>
              <mtd>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>if&#xA0;</mtext>
                </mstyle>
                <munder>
                  <mo>&#x2211;<!-- ∑ --></mo>
                  <mi>j</mi>
                </munder>
                <msub>
                  <mi>w</mi>
                  <mi>j</mi>
                </msub>
                <msub>
                  <mi>x</mi>
                  <mi>j</mi>
                </msub>
                <mo>&#x2264;<!-- ≤ --></mo>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>&#xA0;threshold</mtext>
                </mstyle>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <mn>1</mn>
              </mtd>
              <mtd>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>if&#xA0;</mtext>
                </mstyle>
                <munder>
                  <mo>&#x2211;<!-- ∑ --></mo>
                  <mi>j</mi>
                </munder>
                <msub>
                  <mi>w</mi>
                  <mi>j</mi>
                </msub>
                <msub>
                  <mi>x</mi>
                  <mi>j</mi>
                </msub>
                <mo>&gt;</mo>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>&#xA0;threshold</mtext>
                </mstyle>
              </mtd>
            </mtr>
          </mtable>
          <mo fence="true" stretchy="true" symmetric="true"></mo>
        </mrow>
      </mtd>
    </mlabeledtr>
  </mtable>
</math>

  bias = -threshold , bias is the value that is used to get to the threashold  by adding it or used to activate the perceptron.

<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">
  <mtable columnalign="right center left" rowspacing="3pt" columnspacing="0 thickmathspace" displaystyle="true">
    <mlabeledtr>
         <mtd>
        <mstyle displaystyle="false" scriptlevel="0">
          <mtext>output</mtext>
        </mstyle>
      </mtd>
      <mtd>
        <mi></mi>
        <mo>=</mo>
      </mtd>
        <mrow>
          <mo>{</mo>
          <mtable columnalign="left left" rowspacing="4pt" columnspacing="1em">
            <mtr>
              <mtd>
                <mn>0</mn>
              </mtd>
              <mtd>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>if&#xA0;</mtext>
                </mstyle>
                <mi>w</mi>
                <mo>&#x22C5;<!-- ⋅ --></mo>
                <mi>x</mi>
                <mo>+</mo>
                <mi>b</mi>
                <mo>&#x2264;<!-- ≤ --></mo>
                <mn>0</mn>
              </mtd>
            </mtr>
            <mtr>
              <mtd>
                <mn>1</mn>
              </mtd>
              <mtd>
                <mstyle displaystyle="false" scriptlevel="0">
                  <mtext>if&#xA0;</mtext>
                </mstyle>
                <mi>w</mi>
                <mo>&#x22C5;<!-- ⋅ --></mo>
                <mi>x</mi>
                <mo>+</mo>
                <mi>b</mi>
                <mo>&gt;</mo>
                <mn>0</mn>
              </mtd>
            </mtr>
          </mtable>
          <mo fence="true" stretchy="true" symmetric="true"></mo>
        </mrow>
      </mtd>
    </mlabeledtr>
  </mtable>
</math>
 

### sigmoid neurons
 Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output

the sigmoid neuron has weights for each input, w1,w2,…, and an overall bias, b. But the output is not 0 or 1. Instead, it's σ(w⋅x+b), where σ is called the _sigmoid function_
They can have as output any real number between 0 and 1, so values such as 0.173… and 0.689… are legitimate outputs

### neural network

input layer -> hidden layer -> output layer
input neurons. -> not io neurons -> output neurons

feedforward neural network : neural networks where the output from one layer is used as input to the next layer. Such networks are called _feedforward_ neural networks . This means there are no loops in the network - information is always fed forward, never fed back.

### gradient and gradient descent
 first lets talk about cost function or loss

  \begin{eqnarray}  C(w,b) \equiv
  \frac{1}{2n} \sum_x \| y(x) - a\|^2.
  \tag{6}\end{eqnarray}

cost is the difference from the output which we want to achieve , example if we want output neuron to be 1 but it is 0 then cost is 1

to minimise the cost , gradient descent is used.

- **Gradient** ∇C\nabla C points in the direction where the cost **increases fastest**.
- **Negative gradient** −∇C-\nabla C points in the direction where the cost **decreases fastest**.
- **Gradient descent** is the process of repeatedly moving in that negative-gradient direction.

Lets take an example of ball and slope 
lets put the ball at minimum distance from the end of slope to make sure that it will definetly roll down to the end of slope .

lets represent weight and biases as v

$∇C≡(∂C/∂v1,∂C/∂v2)T.$ 

ΔC≈∇C⋅Δv 

Δv=−η∇C , η is a small positive parameter (known as the _learning rate_).

to start this example lets put the ball at top , it doesnt go down to slope ? lets move it little bit further near the downward slope , do this until it is sure that ball will go downwards ,

v→v′=v−η∇C

one thing to note , we dont want high learning rate since then ball can bounce the slope or not even slow learning rate else it will crawl

- SGD , Stochastic gradient descent

instead of using entire dataset to calculate gradient , use small random group of examples called mini batch

for each input network has its own cost  and totaal cost is the avg of all those each cost, 
to compute the gradient ∇C we need to compute the gradients ∇Cx separately for each training input x
, and then average them,
  ∇C=1n∑x∇Cx

now instead of doing this , we can randmoly pick training examples for small number m from X1 to Xm
and we calulate avg gradient of that instead of full batch, and its approxiamately equal to full gradient


### code part 

- lets initialise neural network class with base Method

```
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

what this does ? 

### before backpropagation lets revise some numerical things which we learnt here


### backpropagation 