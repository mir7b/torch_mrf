# Welcome to Pytorch Markov Randem Fields
This package provides a vecotrized implementation of Markov Random Fields as defined by Daniel Nyga.
The vecotrization is implemented such that every world is processed in parallel. The grounding of the MRF is not vectorizable.

The package heavily relies on PyTorch and is GPU capable. 
When training a Markov Random Field the procedure is just like the Neural Network procedure. You only have to do 3 additional things:

1. When calling loss.backward() set the argument retain_graph=True
2. Clip the weights to be >= 0 with mrf.clip_weights() after each optimizer.step(). This is optional in such a way that the definition of MRFs requires the clique potential to be greater or equal to 0. If you chose to not clip the weights the training will still converge.  
3. Recalculate Z after each optimizer.step() and weight clipping using mrf.calculate_z().

For the training complete world observations should be passed to the Markov Random Field.
The probability of a fully described world can be inferred with the forward method (just like in any other nn.Module).
The probability of a partially described world can be inferred with mrf.predict().


An example on how to feed data to the MRF and train it is given in unit_test.py.