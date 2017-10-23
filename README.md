# Deeplearning-DTU
Autoencoders for Raman sectroscopy

We need to figure out the following:
  How to avoid ending in a local minima with activations being 0.05 plus/minus a small value:
    Vary hyper parameters (batch_size, learning_rate, beta, reg_scale)
    Implement weight warm up / Free bit
  
  Use L-BFGS as optimizer
    https://github.com/tensorflow/tensorflow/issues/446
    https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface
