# Deeplearning-DTU
Autoencoders for Raman sectroscopy

We need to figure out the following:

  How to avoid ending in a local minima with activations being 0.05 plus/minus a small value:
  
    Vary hyper parameters (batch_size, learning_rate, beta, reg_scale)
    
    Implement weight warm up / Free bit (JSL: Done)
  
  Use L-BFGS as optimizer (JT: Done)
    https://github.com/tensorflow/tensorflow/issues/446 / 
    https://www.tensorflow.org/api_docs/python/tf/contrib/opt/ScipyOptimizerInterface

	
Auto Encoder is now implemented in the function Sparse_Non_Neg_AE with required inputs x_train and x_valid, besides that a number of parameters can be passed to the function (see AE_fun.py)
Automatic stop after extra_epoch epochs is implemented. I tried for both Cross Entropy and Least Squares loss, it seems to work quite nice! (see plots)