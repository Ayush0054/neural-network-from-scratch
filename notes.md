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
 

 lets see an example


### sigmoid neurons
 Sigmoid neurons are similar to perceptrons, but modified so that small changes in their weights and bias cause only a small change in their output

the sigmoid neuron has weights for each input, w1,w2,…, and an overall bias, b. But the output is not 0 or 1. Instead, it's σ(w⋅x+b), where σ is called the _sigmoid function_
They can have as output any real number between 0 and 1, so values such as 0.173… and 0.689… are legitimate outputs

### neural network

input layer -> hidden layer -> output layer
input neurons. -> not io neurons -> output neurons

feedforward neural network : neural networks where the output from one layer is used as input to the next layer. Such networks are called _feedforward_ neural networks . This means there are no loops in the network - information is always fed forward, never fed back.

### gradient and gradient descent

### code part 

### backpropagation