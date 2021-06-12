This package provides a vecotrized implementation of Markov Random Fields as defined by Daniel Nyga.
The package heavily relies on PyTorch and is GPU capable. 
When training a Markov Random Field the procedure is just like the Neural Network procedure. You only have to do 3 additional things:

1. When calling loss.backward() set the argument retain_graph=True
2. Clip the weights to be >= 0 with mrf.clip_weights()
3. Recalculate Z after each optimizer.step() using mrf.calculate_z()

An example on how to feed data to the MRF and train it is given in unit_test.py.