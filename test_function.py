# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 08:49:28 2017

@author: jasla
"""

import os
path = "C:\\Users\\jasla\\Dropbox\\phd\\Kurser\\02456 Deep Learning\\Project\\python"

os.chdir(path)


#%% load libraries
from __future__ import division, print_function
import AE_fun as AE
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','nbagg')
get_ipython().run_line_magic('matplotlib','inline')
#import sklearn.datasets
import tensorflow as tf
from tensorflow import layers
#from tensorflow.python.framework.ops import reset_default_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle

# In[2]:

def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    return sig



#%%

# To speed up training we'll only work on a subset of the data containing only the numbers 0, 1.
data = np.load('mnist.npz')
# Possible classes
classes = list(range(10))
# Set classes used, starting from 0.
#included_classes = [0, 1, 4, 9] 
#included_classes = [0, 1, 4, 9, 5, 8]
included_classes = [0,1,2,3,4,5,6,7,8,9]
idxs_train = []
idxs_valid = []
idxs_test = []
num_classes = 0
for c in included_classes:
    if c in classes:
        num_classes += 1
        idxs_train += np.where(data['y_train'] == c)[0].tolist()
        idxs_valid += np.where(data['y_valid'] == c)[0].tolist()
        idxs_test += np.where(data['y_test'] == c)[0].tolist()

print("Number of classes included:", num_classes)
#x_train = data['X_train'][idxs_train].astype('float32')
x_train = data['X_train'].astype('float32')
#targets_train = data['y_train'][idxs_train].astype('int32')
targets_train = data['y_train'].astype('int32')

x_train, targets_train = shuffle(x_train, targets_train, random_state=1234)


#x_valid = data['X_valid'][idxs_valid].astype('float32')
#targets_valid = data['y_valid'][idxs_valid].astype('int32')
x_valid = data['X_valid'].astype('float32')
targets_valid = data['y_valid'].astype('int32')

#x_test = data['X_test'][idxs_test].astype('float32')
#targets_test = data['y_test'][idxs_test].astype('int32')
x_test = data['X_test'].astype('float32')
targets_test = data['y_test'].astype('int32')

print("training set dim(%i, %i)." % x_train.shape)
print("validation set dim(%i, %i)." % x_valid.shape)
print("test set dim(%i, %i)." % x_test.shape)


#%%plot a few MNIST examples
if False:
    idx = 0
    canvas = np.zeros((28*10, 10*28))
    for i in range(10):
        for j in range(10):
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_train[idx].reshape((28, 28))
            idx += 1
    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(canvas, cmap='gray')
    plt.title('MNIST handwritten digits')

#%% Train first Auto Encoder
if False:
    out = AE.Sparse_Non_Neg_AE(x_train = x_train, x_valid = x_valid,use_LS = True,num_epochs = 1000
#                               , train_thresh = 784*0.0194
                               ,train_thresh = 10
                               , n_weight_burn = 5
                               , extra_epoch = 300
#                               ,log_start = -2
                               ,weights_burn_in = np.linspace(0.1,1,num = 10)
                               )
    
    sess,train_loss,train_loss_pure,valid_loss = tuple(out)
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_1 = tuple(sess.run(params))
    saver = tf.train.Saver()
    saver.save(sess, 'models/AE_1_new_loss_v2.ckpt')
    tf.reset_default_graph()
else:
#    sess = tf.Session()
    saver = tf.train.import_meta_graph("models/AE_1_new_loss_v2.ckpt.meta")
    
    with tf.Session() as sess:
        saver.restore(sess,"models/AE_1_new_loss_v2.ckpt")
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_1 = tuple(sess.run(params))
    tf.reset_default_graph()

#%% Evaluate first autoencoder
enc_train = sigmoid(np.matmul(a=x_train,b = param_1[0]) + param_1[1])
enc_valid = sigmoid(np.matmul(a=x_valid,b = param_1[0]) + param_1[1])
enc_test = sigmoid(np.matmul(a=x_test,b = param_1[0]) + param_1[1])

#%% check latent space

tsne = TSNE(n_components=2)
eval_z = tsne.fit_transform(enc_valid)
#pca = PCA(n_components=2)
#eval_z = pca.fit_transform(enc_valid)

plt.figure(figsize=(12, 12))
plt.cla()
plt.title('Latent space')
plt.xlabel('z0'), plt.ylabel('z1')
color = iter(plt.get_cmap('brg')(np.linspace(0, 1.0, num_classes)))
legend_handles = []
for i, c in enumerate(included_classes):
    clr = next(color)
    h = plt.scatter(eval_z[targets_valid==c, 0], eval_z[targets_valid==c, 1], c=clr, s=5., lw=0, marker='o', )
    legend_handles.append(h)
plt.grid('on')
plt.legend(legend_handles, included_classes)

#plt.savefig('latent_space_LS_corrected_cost_164_epoch.png')
#plt.savefig('latent_space_LS_corrected_cost_413_epoch.png')

#%%
plt.figure(figsize=(12, 6))
plt.subplot(2,2,1)

plt.xlabel('Hidden unit'), plt.ylabel('Max activation')
plt.plot(np.max(enc_valid,0), color="black")

plt.grid('on')
plt.subplot(2,2,2)
#plt.title('Error')
plt.xlabel('Test image'), plt.ylabel('Max activation')
plt.plot(np.max(enc_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')
plt.subplot(2,2,3)
plt.xlabel('Hidden unit'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_valid,0), color="black")
plt.grid('on')
plt.subplot(2,2,4)
plt.xlabel('Test image'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')

#plt.savefig('Statistics_activations_LS_corrected_cost_164_epoch.png')
#plt.savefig('Statistics_activations_LS_corrected_cost_413_epoch.png')

#%%Check encoding parameters
encode_weights = param_1[0]
encode_weights_t = encode_weights.transpose()
plt.figure(figsize=(12, 12))
plt.cla()
plt.axis('off')
canvas = np.zeros((28*14, 14*28))
idx = 0
for i in range(14):
    for j in range(14):
        tmp = encode_weights_t[idx]
        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = (tmp).reshape(28,28)
        idx += 1
        
#for i in range(28):
#    for j in range(28):
#        tmp = encode_weights[idx].reshape(14,14)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = (tmp)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = np.log(tmp)
#        idx += 1

cax = plt.imshow(canvas,cmap = 'gray')
cbar = plt.colorbar(cax)

#plt.savefig('Encoding_weights_LS_corrected_cost_164_epoch.png')
#plt.savefig('Encoding_weights_LS_corrected_cost_413_epoch.png')

#%% Check decode parameters
decode_weights = param_1[2].transpose()
decode_weights_t = decode_weights.transpose()
plt.figure(figsize=(12, 12))
plt.cla()
plt.axis('off')
canvas = np.zeros((28*14, 14*28))
idx = 0
for i in range(14):
    for j in range(14):
        tmp = decode_weights_t[idx]
        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = (tmp).reshape(28,28)
        idx += 1
        
#for i in range(28):
#    for j in range(28):
#        tmp = encode_weights[idx].reshape(14,14)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = (tmp)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = np.log(tmp)
#        idx += 1

cax = plt.imshow(canvas,cmap = 'gray')
cbar = plt.colorbar(cax)

#plt.savefig('Decoding_weights_LS_corrected_cost_164_epoch.png')
#plt.savefig('Decoding_weights_LS_corrected_cost_413_epoch.png')

#%%Train second Auto Encoder
if False:
    hiddenSizeL2 = 20;
    out2 = AE.Sparse_Non_Neg_AE(x_train = enc_train, x_valid = enc_valid,use_LS = True,
                                num_epochs = 2000,num_hidden = hiddenSizeL2
#                                ,tau = 0.0001
#                                ,batch_size = 50000
                                ,p_target = 0.05
#                                ,train_thresh = 0.0194 * enc_train.shape[1]
                                ,train_thresh = 2
                                ,epoch_burn_in = 10
                                ,extra_epoch = 300
                                ,weights_burn_in = np.linspace(0.1,1,num = 10)
#                                ,modelpath = "models/AE_2.ckpt"
                                )
    
    sess2,train_loss2,train_loss_pure2,valid_loss2 = tuple(out2)
    params2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_2 = tuple(sess2.run(params2))
    saver = tf.train.Saver()
    saver.save(sess2, 'models/AE_2_new_loss_v2.ckpt')
    tf.reset_default_graph()
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/AE_2_new_loss_v2.ckpt.meta")
    with tf.Session() as sess2:
        saver.restore(sess2,"models/AE_2_new_loss_v2.ckpt")
        params2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_2 = tuple(sess2.run(params2))
    tf.reset_default_graph()
        
#%% Evaluate first autoencoder


enc_2_train = sigmoid(np.matmul(a=enc_train,b = param_2[0]) + param_2[1])
enc_2_valid = sigmoid(np.matmul(a=enc_valid,b = param_2[0]) + param_2[1])
enc_2_test = sigmoid(np.matmul(a=enc_test,b = param_2[0]) + param_2[1])



#%% check latent space

tsne = TSNE(n_components=2)
eval_z = tsne.fit_transform(enc_2_valid)
#pca = PCA(n_components=2)
#eval_z = pca.fit_transform(enc_2_valid)

plt.figure(figsize=(12, 12))
plt.cla()
plt.title('Latent space')
plt.xlabel('z0'), plt.ylabel('z1')
color = iter(plt.get_cmap('brg')(np.linspace(0, 1.0, num_classes)))
legend_handles = []
for i, c in enumerate(included_classes):
    clr = next(color)
    h = plt.scatter(eval_z[targets_valid==c, 0], eval_z[targets_valid==c, 1], c=clr, s=5., lw=0, marker='o', )
    legend_handles.append(h)
plt.grid('on')
plt.legend(legend_handles, included_classes)

#plt.savefig('latent_space AE2_LS_corrected_cost_418_epoch.png')

#%%
plt.figure(figsize=(12, 6))
plt.subplot(2,2,1)

plt.xlabel('Hidden unit'), plt.ylabel('Max activation')
plt.plot(np.max(enc_2_valid,0), color="black")

plt.grid('on')
plt.subplot(2,2,2)
#plt.title('Error')
plt.xlabel('Test image'), plt.ylabel('Max activation')
plt.plot(np.max(enc_2_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')
plt.subplot(2,2,3)
plt.xlabel('Hidden unit'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_2_valid,0), color="black")
plt.grid('on')
plt.subplot(2,2,4)
plt.xlabel('Test image'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_2_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')

#plt.savefig('Statistics_activations_AE2_LS_corrected_cost_418_epoch.png')

#%%Check encoding parameters
encode_weights_2 = param_2[0]
encode_weights_2_t = encode_weights_2.transpose()
plt.figure(figsize=(12, 12))
plt.cla()
plt.axis('off')
canvas = np.zeros((14*5, 14*4))
idx = 0
for i in range(5):
    for j in range(4):
        tmp = encode_weights_2_t[idx]
        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = (tmp).reshape(14,14)
        idx += 1
        
#for i in range(28):
#    for j in range(28):
#        tmp = encode_weights[idx].reshape(14,14)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = (tmp)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = np.log(tmp)
#        idx += 1

cax = plt.imshow(canvas,cmap = 'gray')
cbar = plt.colorbar(cax)

#plt.savefig('Encoding_weights_AE2_LS_corrected_cost_418_epoch.png')

#%% Check decode parameters
decode_weights_2 = param_2[2].transpose()
decode_weights_2_t = decode_weights_2.transpose()
plt.figure(figsize=(12, 12))
plt.cla()
plt.axis('off')
canvas = np.zeros((5*14, 14*4))
idx = 0
for i in range(5):
    for j in range(4):
        tmp = decode_weights_2_t[idx]
        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = (tmp).reshape(14,14)
        idx += 1
        
#for i in range(28):
#    for j in range(28):
#        tmp = encode_weights[idx].reshape(14,14)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = (tmp)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = np.log(tmp)
#        idx += 1

cax = plt.imshow(canvas,cmap = 'gray')
cbar = plt.colorbar(cax)

#plt.savefig('Decoding_weights_AE2_LS_corrected_cost_418_epoch.png')

#%%
#y_train = tf.one_hot(indices = targets_train,depth = 10)
#y_valid = tf.one_hot(indices = targets_valid,depth = 10)
#y_test = tf.one_hot(indices = targets_test,depth = 10)
y_train = np.eye(10)[targets_train]
y_valid = np.eye(10)[targets_valid]
y_test = np.eye(10)[targets_test]
alpha = 0.003


#%%Train softmax layer
tf.reset_default_graph()
if True:
#    out3 = AE.Non_neg_softmax(x_train = enc_2_train,y_train = y_train, x_valid = enc_2_valid,y_valid = y_valid,
    out3 = AE.Non_neg_softmax(x_train = enc_train,y_train = y_train, x_valid = enc_valid,y_valid = y_valid,
                                num_epochs = 400
#                                ,tau = 0.0001
#                                ,batch_size = 50
                                ,batch_size = 1000
#                                ,train_thresh = 0.0194 * enc_train.shape[1]
#                                ,train_thresh = 2
#                                ,modelpath = "models/AE_2.ckpt"
                                )
    
    sess3,train_loss3,train_acc3,valid_loss3,valid_acc3 = tuple(out3)
    params3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_3 = tuple(sess3.run(params3))
    saver = tf.train.Saver()
    saver.save(sess2, 'models/Softmax_new_loss_v2.ckpt')
    tf.reset_default_graph()
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/Softmax_new_loss_v2.ckpt.meta")
    with tf.Session() as sess3:
        saver.restore(sess3,"models/Softmax_new_loss_v2.ckpt")
        params3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_3 = tuple(sess3.run(params3))
#%% Define softmax layer
tf.reset_default_graph()
x_pl = tf.placeholder(tf.float32, [None, hiddenSizeL2], 'x_pl')
y_pl = tf.placeholder(tf.float32, [None, num_classes], name='yPlaceholder')
l_softmax = layers.dense(inputs=x_pl, units=num_classes, activation=tf.nn.softmax, name='l_softmax')

y = l_softmax

#%% Define softmax cost
cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+1e-8), reduction_indices=[1])

# averaging over samples
cross_entropy = tf.reduce_mean(cross_entropy)

loss_softmax = cross_entropy

params_3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

gradients_softmax = tf.gradients(loss_softmax,params_3)
gradients_softmax[0] -= alpha * tf.nn.relu(-params_3[0])
train_op = optimizer.apply_gradients(zip(gradients_softmax,params))

correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))

# averaging the one-hot encoded vector
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#%% Test forward pass
_x_test = np.zeros(shape=(32, hiddenSizeL2))
# initialize the Session
sess = tf.Session()

sess.run(tf.global_variables_initializer())
        
feed_dict = {x_pl: _x_test}
res_forward_pass = sess.run(fetches=[l_softmax], feed_dict=feed_dict)
print("l_out", res_forward_pass[0].shape)

#%% Train network
num_samples_train = x_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    #updates = []
    
    train_loss = []
    train_loss_pure = []
    train_sparse = []
    train_reg = []
    valid_loss = []


