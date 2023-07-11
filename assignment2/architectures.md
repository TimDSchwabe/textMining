Home Assignment 2

Tim Schwabe, Erik Vogel, Mark Nagengast Porro


Task 1:
For this exercise, two architectures were constructed.
An ordinary RNN, which takes a sequence and embeds every word within it. On it, a Gated Recurrent Unit (GRU) is executed. The last hidden state of the i-th sequence is used as the inital hidden state of the (i+1)-th sequence, which is done by using two seperate GRUs.
The other architecture uses additive attention. The tokens are being iterated over one by one, while only ever considering past tokens, not subsequent ones.

Task 2:
Here, we used a single ordinary RNN. The initial hidden state is set with random values, as was done in the exercise.