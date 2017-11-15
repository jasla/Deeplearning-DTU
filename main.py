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
from tensorflow.python.ops.nn import relu, sigmoid, tanh
#from tensorflow.python.framework.ops import reset_default_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import itertools
# In[2]:

def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    return sig

def softmax(x):
#    expx = [np.exp(x) for x in X]
    expx = np.exp(x.T - np.max(x,1)).T # avoid numerically instability
#    expx = np.exp(x)
    soft_max = (expx.T * (1/ np.sum(expx,1))).T
    return(soft_max)
#    return(expx)


#softmax(np.array([[1,1,2,0,10,1],[0,9,5,3,4,6]]))
#Out[106]: 
#array([[  1.23317182e-04,   1.23317182e-04,   3.35210854e-04,
#          4.53658558e-05,   9.99249472e-01,   1.23317182e-04],
#       [  1.14539540e-04,   9.28123502e-01,   1.69991749e-02,
#          2.30058815e-03,   6.25364697e-03,   4.62085483e-02]])
#%%

# To speed up training we'll only work on a subset of the data containing only the numbers 0, 1.
data = np.load('mnist.npz')
# Possible classes
classes = list(range(10))
classes_plot = classes
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
tf.reset_default_graph()
if False:
    hiddenSizeL2 = 20;
    out2 = AE.Sparse_Non_Neg_AE(x_train = enc_train, x_valid = enc_valid,use_LS = True,
                                num_epochs = 2000,num_hidden = hiddenSizeL2
#                                ,tau = 0.0001
#                                ,batch_size = 300
                                ,batch_size = 1000
                                ,p_target = 0.05
#                                ,train_thresh = 0.0194 * enc_train.shape[1]
                                ,train_thresh = 2
                                ,epoch_burn_in = 30
                                ,extra_epoch = 2000
#                                ,weights_burn_in = np.linspace(0.1,1,num = 30)
                                ,n_weight_burn = 30
#                                ,modelpath = "models/AE_2.ckpt"
                                )
    
    sess2,train_loss2,train_loss_pure2,valid_loss2,train_reg2,train_sparse2 = tuple(out2)
    params2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_2 = tuple(sess2.run(params2))
    saver = tf.train.Saver()
    saver.save(sess2, 'models/AE_2_new_loss_v6.ckpt')
    tf.reset_default_graph()
    
    #v4: batch_size = 1000, ,n_weight_burn = 30, epoch_burn_in = 30, extra_epoch = 300
    #v4 final line: Epoch 1203. Train: 0.42857. Pure Train: 0.41774. Val: 0.42319. Sparse: 0.00813. Sparse Weight: 1.00000 Weight: 0.00270
    #v5: num_epochs = 2000,batch_size = 300,train_thresh = 2,epoch_burn_in = 10,extra_epoch = 2000 ,n_weight_burn = 30
    #v5, final line: Epoch 2000. Train: 0.34745. Pure Train: 0.33493. Val: 0.33618. Sparse: 0.00106. Sparse Weight: 1.00000 Weight: 0.01147 ## sparse achieved by driving activations down
    #v6: num_epochs = 2000,batch_size = 1000,train_thresh = 2,epoch_burn_in = 30,extra_epoch = 2000 ,n_weight_burn = 30
    #v6, final line: Epoch 2000. Train: 0.40001. Pure Train: 0.38750. Val: 0.38886. Sparse: 0.00645. Sparse Weight: 1.00000 Weight: 0.00606
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/AE_2_new_loss_v5.ckpt.meta")
    with tf.Session() as sess2:
        saver.restore(sess2,"models/AE_2_new_loss_v5.ckpt")
        params2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_2 = tuple(sess2.run(params2))
    tf.reset_default_graph()
        
#%% Evaluate second autoencoder


enc_2_train = sigmoid(np.matmul(a=enc_train,b = param_2[0]) + param_2[1])
enc_2_valid = sigmoid(np.matmul(a=enc_valid,b = param_2[0]) + param_2[1])
enc_2_test = sigmoid(np.matmul(a=enc_test,b = param_2[0]) + param_2[1])

#%% Plot errors

updates = [i*x_train.shape[0] for i in range(1,np.shape(train_loss2)[0]+1)]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.semilogy(updates,train_loss2,color="black")
plt.semilogy(updates,train_loss_pure2,color="red")
plt.semilogy(updates,valid_loss2,color="gray")
plt.legend(['Train Error','Pure error', 'Valid Error'])
plt.subplot(1,2,2)
plt.semilogy(updates,train_reg2,color="gray")
plt.semilogy(updates,train_sparse2,color="black")
plt.legend(["Weight decay","Sparseness"])

#plt.savefig('Training_error_AE_2_v6.png')





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
#plt.savefig('latent_space AE2_v6.png')
#plt.savefig('latent_space AE2_v5.png')

#%%
plt.figure(figsize=(12, 6))
plt.subplot(2,2,1)

plt.xlabel('Hidden unit'), plt.ylabel('Max activation')
plt.plot(np.max(enc_2_valid,0), color="black")

plt.grid('on')
plt.subplot(2,2,2)
#plt.title('Error')
plt.xlabel('Validation image'), plt.ylabel('Max activation')
plt.plot(np.max(enc_2_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')
plt.subplot(2,2,3)
plt.xlabel('Hidden unit'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_2_valid,0), color="black")
plt.grid('on')
plt.subplot(2,2,4)
plt.xlabel('Validation image'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_2_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')

#plt.savefig('Statistics_activations_AE2_LS_corrected_cost_418_epoch.png')
#plt.savefig('Statistics_activations_AE2_v6.png')
#plt.savefig('Statistics_activations_AE2_v5.png')

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
        

cax = plt.imshow(canvas,cmap = 'gray')
cbar = plt.colorbar(cax)

#plt.savefig('Decoding_weights_AE2_LS_corrected_cost_418_epoch.png')

#%%
y_train = np.eye(10)[targets_train]
y_valid = np.eye(10)[targets_valid]
y_test = np.eye(10)[targets_test]
alpha = 0.003


#%%Train softmax layer
tf.reset_default_graph()
if False:
    out3 = AE.Non_neg_softmax(x_train = enc_2_train,y_train = y_train, x_valid = enc_2_valid,y_valid = y_valid,
#    out3 = AE.Non_neg_softmax(x_train = enc_train,y_train = y_train, x_valid = enc_valid,y_valid = y_valid,
                                num_epochs = 400
#                                ,tau = 0.0001
#                                ,batch_size = 50
                                ,batch_size = 300
#                                ,train_thresh = 0.0194 * enc_train.shape[1]
#                                ,train_thresh = 2
#                                ,modelpath = "models/Softmax_new_loss_v4.ckpt"
                                )
    
    sess3,train_loss3,train_acc3,valid_loss3,valid_acc3 = tuple(out3)
    params3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_3 = tuple(sess3.run(params3))
    saver = tf.train.Saver()
    saver.save(sess3, 'models/Softmax_new_loss_v4.ckpt')
    tf.reset_default_graph()
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/Softmax_new_loss_v4.ckpt.meta")
    with tf.Session() as sess3:
        saver.restore(sess3,"models/Softmax_new_loss_v4.ckpt")
        params3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_3 = tuple(sess3.run(params3))
        
#%% Fine tune
# only do 100 epoch
tf.reset_default_graph()
if False:
    out4 = AE.Sparse_Non_Neg_Stacked(x_train = x_train, x_valid = x_valid, 
                              y_train = y_train, y_valid = y_valid, 
                              params = [param_1,param_2,param_3], num_epochs = 100)
    sess4, train_loss4, train_acc4, valid_loss4, valid_acc4, train_reg4, train_sparse4 = tuple(out4)
    params4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_4 = tuple(sess4.run(params4))
    saver = tf.train.Saver()
    saver.save(sess4, 'models/Finetune_v1.ckpt')
    tf.reset_default_graph()
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/Finetune_v1.ckpt.meta")
    with tf.Session() as sess3:
        saver.restore(sess3,"models/Finetune_v1.ckpt")
        params4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_4 = tuple(sess3.run(params3))
#%% Evaluate all layers
enc_fine_train = sigmoid(np.matmul(a=x_train,b = param_4[0]) + param_4[1])
enc_fine_valid = sigmoid(np.matmul(a=x_valid,b = param_4[0]) + param_4[1])
enc_fine_test = sigmoid(np.matmul(a=x_test,b = param_4[0]) + param_4[1])

enc_2_fine_train = sigmoid(np.matmul(a=enc_fine_train,b = param_4[2]) + param_4[3])
enc_2_fine_valid = sigmoid(np.matmul(a=enc_fine_valid,b = param_4[2]) + param_4[3])
enc_2_fine_test = sigmoid(np.matmul(a=enc_fine_test,b = param_4[2]) + param_4[3])

yhat_train = softmax(np.matmul(a = enc_2_fine_train, b = param_4[4]) + param_4[5])
yhat_valid = softmax(np.matmul(a = enc_2_fine_valid, b = param_4[4]) + param_4[5])
yhat_test = softmax(np.matmul(a = enc_2_fine_test, b = param_4[4]) + param_4[5])

preds_train = np.argmax(yhat_train,axis=1)
preds_valid = np.argmax(yhat_valid,axis=1)
preds_test = np.argmax(yhat_test,axis=1)
#%% 
updates = [i*x_train.shape[0] for i in range(1,np.shape(train_loss4)[0]+1)]
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.semilogy(updates,train_loss4,color="black")
plt.semilogy(updates,valid_loss4,color="gray")
plt.legend(['Train Error', 'Valid Error'])

plt.subplot(3,1,2)
plt.semilogy(updates,train_acc4,color="black")
plt.semilogy(updates,valid_acc4,color="gray")
plt.legend(['Train Acc', 'Valid Acc'])

plt.subplot(3,1,3)
plt.semilogy(updates,train_reg4,color="gray")
plt.semilogy(updates,train_sparse4,color="black")
plt.legend(["Weight decay","Sparseness"])
#plt.savefig('Training_error_finetune.png')

#%% Check latent spaces
tsne = TSNE(n_components=2)
hidden_1_finetune = tsne.fit_transform(enc_fine_valid)
hidden_2_finetune = tsne.fit_transform(enc_2_fine_valid)
#pca = PCA(n_components=2)
#eval_z = pca.fit_transform(enc_2_valid)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.cla()
plt.title('First layer Latent space')
plt.xlabel('z0'), plt.ylabel('z1')
color = iter(plt.get_cmap('brg')(np.linspace(0, 1.0, num_classes)))
legend_handles = []
for i, c in enumerate(included_classes):
    clr = next(color)
    h = plt.scatter(hidden_1_finetune[targets_valid==c, 0], hidden_1_finetune[targets_valid==c, 1], c=clr, s=5., lw=0, marker='o', )
    legend_handles.append(h)
plt.grid('on')
plt.legend(legend_handles, classes_plot)

plt.subplot(1,2,2)
plt.cla()
plt.title('Second layer Latent space')
plt.xlabel('z0'), plt.ylabel('z1')
color = iter(plt.get_cmap('brg')(np.linspace(0, 1.0, num_classes)))
legend_handles = []
for i, c in enumerate(included_classes):
    clr = next(color)
    h = plt.scatter(hidden_2_finetune[targets_valid==c, 0], hidden_2_finetune[targets_valid==c, 1], c=clr, s=5., lw=0, marker='o', )
    legend_handles.append(h)
plt.grid('on')
plt.legend(legend_handles, classes_plot)
#plt.savefig("latent_space_finetune.png")

#%% Check activations first layer
plt.figure(figsize=(12, 6))
plt.subplot(2,2,1)
plt.xlabel('Hidden unit'), plt.ylabel('Max activation')
plt.plot(np.max(enc_fine_valid,0), color="black")

plt.grid('on')
plt.subplot(2,2,2)

plt.xlabel('Validation image'), plt.ylabel('Max activation')
plt.plot(np.max(enc_fine_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')
plt.subplot(2,2,3)
plt.xlabel('Hidden unit'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_fine_valid,0), color="black")
plt.grid('on')
plt.subplot(2,2,4)
plt.xlabel('Validation image'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_fine_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')

#plt.savefig("Statistics_activations_finetune_l1.png")

#%% Check activations second layer
plt.figure(figsize=(12, 6))
plt.subplot(2,2,1)
plt.xlabel('Hidden unit'), plt.ylabel('Max activation')
plt.plot(np.max(enc_2_fine_valid,0), color="black")

plt.grid('on')
plt.subplot(2,2,2)

plt.xlabel('Validation image'), plt.ylabel('Max activation')
plt.plot(np.max(enc_2_fine_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')
plt.subplot(2,2,3)
plt.xlabel('Hidden unit'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_2_fine_valid,0), color="black")
plt.grid('on')
plt.subplot(2,2,4)
plt.xlabel('Validation image'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_2_fine_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')
#plt.savefig("Statistics_activations_finetune_l2.png")

#%% First encoding layer
encode_weights_final = param_4[0]
encode_weights_t = encode_weights_final.transpose()
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
        
cax = plt.imshow(canvas,cmap = 'gray')
cbar = plt.colorbar(cax)

#plt.savefig('Encoding_weights_Fine_tune_layer_1.png')

#%% Second encoding layer
encode_weights_final_2 = param_4[2]
encode_weights_final_2_t = encode_weights_final_2.transpose()
plt.figure(figsize=(12, 12))
plt.cla()
plt.axis('off')
canvas = np.zeros((14*5, 14*4))
idx = 0
for i in range(5):
    for j in range(4):
        tmp = encode_weights_final_2_t[idx]
        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = (tmp).reshape(14,14)
        idx += 1
        


cax = plt.imshow(canvas,cmap = 'gray')
cbar = plt.colorbar(cax)

#plt.savefig('Encoding_weights_Fine_tune_layer_2.png')

#%% Confusion matrices
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

cnf_valid = confusion_matrix(preds_valid,targets_valid); cnf_valid = cnf_valid.astype('float') / cnf_valid.sum(axis=1)[:, np.newaxis]
cnf_test = confusion_matrix(preds_test,targets_test); cnf_test = cnf_test.astype('float') / cnf_test.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plot_confusion_matrix(cnf_valid, classes=classes_plot, normalize=True,
                      title='Confusion matrix, validation data')
plt.subplot(1,2,2)
plot_confusion_matrix(cnf_test, classes=classes_plot, normalize=True,
                      title='Confusion matrix, test data')

#plt.savefig("Confusion_matrix_fine_tune.png")
