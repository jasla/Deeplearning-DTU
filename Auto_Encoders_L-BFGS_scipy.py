from __future__ import division, print_function
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle

from tensorflow import layers
from tensorflow.contrib.layers import fully_connected 
from tensorflow.python.ops.nn import relu, sigmoid, tanh

from utils import setup_data, plot_mnist_ex

LOAD_MODEL = False

SCIPY_LOSS_TRAIN = list()
ITERATION = 0
MAXITER = 400

def run():
    x_train, x_valid, x_test, targets_train, targets_valid, targets_test, num_classes, included_classes = setup_data()

    plot_mnist_ex(x_train)

    setup_model_scipy(x_train, x_valid, x_test, targets_train, targets_valid, targets_test)


def save_fetch(loss, l_out,loss_pure,reg_sparse,reg_term):
    global ITERATION, SCIPY_LOSS_TRAIN
    ITERATION += 1
    print('Iteration {:d}. Train: {:f}.'.format(ITERATION,loss[0][0]))
    SCIPY_LOSS_TRAIN.append(loss[0][0])



def setup_model_scipy(x_train, x_valid, x_test, targets_train, targets_valid, targets_test):

    #Scipy uses a session-based optimizer.
    # define in/output size
    num_features = x_train.shape[1]
    num_hidden_enc = 196
    num_hidden_dec = 128

    x_pl = tf.placeholder(tf.float32, [None, num_features], 'x_pl')

    l_enc = layers.dense(inputs=x_pl, units=num_hidden_enc, activation=sigmoid, name='l_enc')
    l_out = layers.dense(inputs=l_enc, units=num_features, activation=sigmoid, name='l_dec')
    ## Define loss function
    eps = 10**(-10)
    loss_per_pixel = - x_pl * tf.log(l_out+eps) - (1 - x_pl) * tf.log(1 - l_out + eps)
    loss_pure = tf.reduce_mean(loss_per_pixel, name="mean_error")

    # If you want regularization
    reg_scale = 0.003
    beta = 3.

    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    reg_term = tf.reduce_sum([tf.reduce_sum(tf.nn.relu(-param)**2) for param in params if param.name.endswith('kernel:0')])
    sparse_param = 0.05

    sparse_pl = tf.placeholder(tf.float32, [1, 1], 'sparse_pl')

    p_act_enc = tf.reduce_mean(l_enc,0)

    reg_sparse = tf.reduce_sum(sparse_param * tf.log(sparse_param/p_act_enc) + (1-sparse_param)*tf.log((1-sparse_param)/(1-p_act_enc)))

    loss = loss_pure + sparse_pl * beta * reg_sparse

    global MAXITER
    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B',options={'maxiter': MAXITER})
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        #%% Train
        fetches_train = [loss, l_out,loss_pure,reg_sparse,reg_term]
        feed_dict_train = {x_pl: x_train, sparse_pl: [[0.02]]}
        optimizer.minimize(session, fetches=fetches_train, feed_dict=feed_dict_train, loss_callback=save_fetch)
           
        global SCIPY_LOSS_TRAIN
        plt.figure(figsize=(7, 7))
        plt.title('Error')
        plt.xlabel('Updates'), plt.ylabel('Error')
        plt.semilogy(SCIPY_LOSS, color="black")
        plt.show(block=True)


if __name__ == "__main__":
    run()