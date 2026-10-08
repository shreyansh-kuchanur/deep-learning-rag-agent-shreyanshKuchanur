# Long Short-Term Memory (LSTM) Networks

## Why LSTMs Were Introduced

Standard recurrent neural networks (RNNs) process a sequence one step at a time and pass a hidden state forward. When they are trained with backpropagation through time, the gradient is multiplied by the same weight matrix at every step. Over long sequences this repeated multiplication makes the gradient shrink towards zero, which is called the vanishing gradient problem. As a result, a plain RNN struggles to learn dependencies between events that are far apart in the sequence. LSTMs, introduced by Hochreiter and Schmidhuber in 1997, were designed to solve this problem.

## The Cell State

The key idea of an LSTM is the cell state, a separate memory line that runs through the whole sequence. Information on the cell state is changed only by small, controlled additions and removals rather than by repeated matrix multiplication. Because the cell state is updated additively, gradients can flow backwards through many time steps without vanishing. This lets the network keep useful information for a long time.

## The Three Gates

An LSTM controls its cell state with three gates. Each gate is a small neural layer with a sigmoid activation that outputs values between 0 and 1, where 0 means "let nothing through" and 1 means "let everything through".

### Forget Gate

The forget gate decides what information to remove from the cell state. It looks at the previous hidden state and the current input and outputs a number between 0 and 1 for each value in the cell state. A value close to 0 tells the LSTM to forget that piece of information, for example forgetting the subject of an old sentence once a new subject appears.

### Input Gate

The input gate decides what new information to store in the cell state. It works together with a tanh layer that creates candidate values. The input gate chooses which of these candidate values are actually written into memory.

### Output Gate

The output gate decides what part of the cell state is exposed as the hidden state for the current time step. The cell state is passed through tanh and multiplied by the output gate, so the network only outputs the information that is relevant right now.

## Common Use Cases

LSTMs are well suited to tasks with long-range dependencies in sequential data. Typical applications include machine translation, speech recognition, text generation, handwriting recognition and time series forecasting such as predicting stock prices or energy demand.

## LSTM vs GRU

The Gated Recurrent Unit (GRU) is a simpler variant of the LSTM. It merges the forget and input gates into a single update gate and combines the cell state and hidden state. GRUs have fewer parameters and often train faster, while LSTMs can perform slightly better on tasks that need very long memory.
