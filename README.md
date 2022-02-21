# Welcome to Markov Random Fields in PyTorch
This package provides a vectorized implementation of Markov Random Fields that is capable of GPU acceleration.

The packages can be installed by navigating to the torch_random_variable folder and calling pip install . there. 
After that pip install . should be called in the directory above. 

The theory behind MRFs is best understood through the book "Probabilistic Graphical Models" by Daphne Koller and Nir Friedmann.
The book can be found for free under this link https://djsaunde.github.io/read/books/pdfs/probabilistic%20graphical%20models.pdf .

In examples/mnist_mn.py an example can be found that shows the interface to the torch_mrf package.

The example performs classification on the MNIST handwritten digit dataset. The structure of the MRF is a logistic regression
like structure.  

# Complexity Discussion
It is commonly known that MRFs are exponential hard due to the calculation of the partition function. To become even worse
the partition function is also part of the derivative of the log-likelihood of an MRF and therefore training was known to be unfeasible until today. This MRF implementation exploits the intuitive explanation of cliques.  Cliques represent the relative 
potential that is associated with a scenario. For example the potentials \phi(A,B) = 5 and \phi(A,!B) = 10 state that A and not B is twice as likely as A and B. Therefore the cliques are optimal if the represent the joint probability distribution of their variables.
The parameters can be found via probabilistic inference on the data. This procedure takes O(N^2 * |\phi|) steps and is therefore suitable for large problems. The calculation Z can be avoided in almost every AI scenario. AI scenarios almost ever contain a question
of which world is more likely. The outputs for the probabilities of the worlds of an MRF are all distorted by the factor 1/Z. This distortion can be ignored in comparing scenarios. 
To conclude this repository provides MRFs that can be fit in quadratic time and where inference is done in linear time. 