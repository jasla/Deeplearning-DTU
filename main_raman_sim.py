# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 09:14:57 2017

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
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import csv
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


#%%
Data = np.zeros(shape = (0,300))
#with open(".\\data\\raman_30x30_wavenumbers300_hotspots10_40dB.csv","r") as f:
with open(".\\data\\raman_30x30_wavenumbers300_hotspots10_0dB.csv","r") as f:
    fileReader = csv.reader(f, delimiter = ",")
    
    for line in fileReader:
        data = []
        tmp = []
        counter = 0
        for val in line:
            tmp += [float(val)]
            counter += 1
            if counter == 900:
                counter = 0
                data += [tmp]
                tmp = []
        data = np.array(data).T
        Data = np.concatenate((Data,data))
        
#        break

#Data = np.array(data)

#with open(".\\data\\raman_30x30_wavenumbers300_hotspots10_40dB_labels.csv","r") as f:
with open(".\\data\\raman_30x30_wavenumbers300_hotspots10_0dB_labels.csv","r") as f:
    fileReader = csv.reader(f, delimiter = ",")
    labels = []
    for line in fileReader:
        labels += [int(val) for val in line]


#targets = np.array([lab for lab in labels for i in range(900)])
targets = np.array([i for i in range(len(labels)) for j in range(900)])
included_classes = [i for i in range(len(labels))]
#included_classes = list(set(labels))
num_classes = len(included_classes)
classes_plot = included_classes
#%%
idx = np.random.randint(low=0,high=Data.shape[0],size = (1000,))
plt.figure(figsize=(12,12))
for i in range(9):
    idx = [j for j in range(i*50,(i+1)*50)]
    ax = plt.subplot(3,3,i+1)
    tmp = ax.imshow(Data[idx,],aspect = 6)

#%%
plt.figure(figsize = (15,15))
count = 0
for k in range(18):
    for j in range(900):
            plt.subplot(9,2,k+1)
            plt.plot(Data[count,])
            count += 1

#plt.savefig("Raman_spectre_sim.png")
#plt.savefig("Raman_spectre_sim_1dB.png")
            
#%%
plt.figure(figsize = (15,15))
#count = 0
counter = 0
for k in [7,15,23,31]:
    count = (k)*900
    counter += 1
    for j in range(900):
            plt.subplot(4,1,counter)
            plt.plot(Data[count,])
            count += 1
            
#%% Investigate class number 8
plt.figure(figsize = (15,15))
#count = 0
counter = 0
for k in [7,8,18]:
    count = (k)*900
    counter += 1
    for j in range(900):
            plt.subplot(3,1,counter)
            plt.plot(Data[count,])
            count += 1
#%%
n_maps = 10
nmf = NMF(n_components = n_maps+2).fit(Data[0:(900*n_maps),])
pred = nmf.fit_transform(Data[0:(900*n_maps),])

#%%
plt.figure(figsize=(12,12))
for i in range(n_maps+2):
    plt.scatter([i]*(900*n_maps),pred[0:(900*n_maps),i])
    
#%%
plt.figure(figsize = (12,12))
for i in range(n_maps+2):
    plt.subplot(6,2,i+1)
    plt.plot(pred[:,i])
    for j in range(n_maps):
        plt.axvline(x = 900*j)
#%%
plt.figure(figsize=(15,12))
count = 0
for i in range(9):
    ax = plt.subplot(9,1,i+1)
    for j in range(5):
        ax.plot(nmf.components_[count,])
        count += 1
#%%
if False:
    #data = np.load('mnist.npz')
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
else:
    idx = [i for i in range(Data.shape[0])]
    idx = np.random.permutation(idx)
    n_train = 30000
    n_valid = 10000
    n_test = 5000
    x_train = Data[idx[:n_train],]
    x_valid = Data[idx[n_train:(n_train+n_valid)],]
    x_test = Data[idx[(n_train+n_valid):],]
    targets_train = targets[idx[:n_train]]
    targets_valid = targets[idx[n_train:(n_train+n_valid)]]
    targets_test = targets[idx[(n_train+n_valid):]]
    print("training set dim(%i, %i)." % x_train.shape)
    print("validation set dim(%i, %i)." % x_valid.shape)
    print("test set dim(%i, %i)." % x_test.shape)


#%% Train first Auto Encoder
tf.reset_default_graph()
if True:
    out = AE.Sparse_Non_Neg_AE(x_train = x_train, x_valid = x_valid,use_LS = True,num_epochs = 1000
                               , num_hidden = 50
#                               , train_thresh = 784*0.0194
                               ,train_thresh = 3
#                               , n_weight_burn = 10
                               , epoch_burn_in = 15
                               , extra_epoch = 300
#                               ,log_start = -2
#                               ,weights_burn_in = np.linspace(0.1,1,num = 10)
                               )
    
    sess,train_loss,train_loss_pure,valid_loss,train_reg,train_sparse = tuple(out)
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_1 = tuple(sess.run(params))
    saver = tf.train.Saver()
    saver.save(sess, 'models/raman_sim_layer_1_2.ckpt')
    tf.reset_default_graph()
else:
#    sess = tf.Session()
    saver = tf.train.import_meta_graph("models/raman_sim_layer_1_2.ckpt.meta")
    
    with tf.Session() as sess:
        saver.restore(sess,"models/raman_sim_layer_1_2.ckpt")
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_1 = tuple(sess.run(params))
    tf.reset_default_graph()

#%% Evaluate first autoencoder
enc_train = sigmoid(np.matmul(a=x_train,b = param_1[0]) + param_1[1])
enc_valid = sigmoid(np.matmul(a=x_valid,b = param_1[0]) + param_1[1])
enc_test = sigmoid(np.matmul(a=x_test,b = param_1[0]) + param_1[1])

#%% Plot errors

updates = [i*x_train.shape[0] for i in range(1,np.shape(train_loss)[0]+1)]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.semilogy(updates,train_loss,color="black")
plt.semilogy(updates,train_loss_pure,color="red")
plt.semilogy(updates,valid_loss,color="gray")
plt.legend(['Train Error','Pure error', 'Valid Error'])
plt.subplot(1,2,2)
plt.semilogy(updates,train_reg,color="gray")
plt.semilogy(updates,train_sparse,color="black")
plt.legend(["Weight decay","Sparseness"])

#plt.savefig('Training_error_raman_sim_layer_1_2.png')
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

#plt.savefig('raman_sim_layer_1_latent_space_1000_epoch.png')
#plt.savefig('raman_sim_layer_1_latent_space_470_epoch.png')

#%%
plt.figure(figsize=(12, 6))
plt.subplot(2,2,1)

plt.xlabel('Hidden unit'), plt.ylabel('Max activation')
plt.plot(np.max(enc_valid,0), color="black")

plt.grid('on')
plt.subplot(2,2,2)
#plt.title('Error')
plt.xlabel('Validation image'), plt.ylabel('Max activation')
plt.plot(np.max(enc_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')
plt.subplot(2,2,3)
plt.xlabel('Hidden unit'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_valid,0), color="black")
plt.grid('on')
plt.subplot(2,2,4)
plt.xlabel('Validation image'), plt.ylabel('Mean activation')
plt.plot(np.mean(enc_valid,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')

#plt.savefig('Statistics_activations_raman_sim_layer_1_1000_epoch_2.png')

#%%Check encoding parameters
encode_weights = param_1[0]
encode_weights_t = encode_weights.transpose()
plt.figure(figsize=(15, 15))
plt.cla()
plt.axis('off')


counter = 0
for i in range(10):
    plt.subplot(10,1,i+1)
    for j in range(5):
        plt.plot(encode_weights[:,counter])
        counter += 1

#plt.savefig('Encoding_weights_raman_sim_layer_1_1000_epoch_2.png')

#%%
plt.figure(figsize=(20,15))
count = 1
for c in included_classes:
    ax=plt.subplot(10,5,count)
    plt.boxplot(enc_valid[targets_valid==c,:])
    count+=1
#plt.savefig('raman_sim_boxplot_layer_1_1000_epoch.png')
#plt.savefig('raman_sim_boxplot_layer_1_470_epoch.png')

#%% Check decode parameters
decode_weights = param_1[2].transpose()
decode_weights_t = decode_weights.transpose()
plt.figure(figsize=(15, 15))
plt.cla()
plt.axis('off')

counter = 0
for i in range(10):
    plt.subplot(10,1,i+1)
    for j in range(5):
        plt.plot(decode_weights[:,counter])
        counter += 1


#cax = plt.imshow(canvas,cmap = 'gray')
#cbar = plt.colorbar(cax)

#plt.savefig('Decoding_weights_LS_corrected_cost_164_epoch.png')
#plt.savefig('Decoding_weights_LS_corrected_cost_413_epoch.png')

#%%Train second Auto Encoder
tf.reset_default_graph()
if False:
    hiddenSizeL2 = 6;
    out2 = AE.Sparse_Non_Neg_AE(x_train = enc_train, x_valid = enc_valid,use_LS = True,
                                num_epochs = 2000,num_hidden = hiddenSizeL2
#                                ,tau = 0.0001
#                                ,batch_size = 300
                                ,batch_size = 1000
                                ,p_target = 1/hiddenSizeL2
#                                ,train_thresh = 0.0194 * enc_train.shape[1]
                                ,train_thresh = 2
                                ,epoch_burn_in = 30
                                ,extra_epoch = 100
#                                ,weights_burn_in = np.linspace(0.1,1,num = 30)
                                ,n_weight_burn = 30
#                                ,modelpath = "models/AE_2.ckpt"
                                )
    
    sess2,train_loss2,train_loss_pure2,valid_loss2,train_reg2,train_sparse2 = tuple(out2)
    params2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_2 = tuple(sess2.run(params2))
    saver = tf.train.Saver()
    saver.save(sess2, 'models/raman_sim_layer_2_1.ckpt')
    tf.reset_default_graph()
    
    #v4: batch_size = 1000, ,n_weight_burn = 30, epoch_burn_in = 30, extra_epoch = 300
    #v4 final line: Epoch 1203. Train: 0.42857. Pure Train: 0.41774. Val: 0.42319. Sparse: 0.00813. Sparse Weight: 1.00000 Weight: 0.00270
    #v5: num_epochs = 2000,batch_size = 300,train_thresh = 2,epoch_burn_in = 10,extra_epoch = 2000 ,n_weight_burn = 30
    #v5, final line: Epoch 2000. Train: 0.34745. Pure Train: 0.33493. Val: 0.33618. Sparse: 0.00106. Sparse Weight: 1.00000 Weight: 0.01147 ## sparse achieved by driving activations down
    #v6: num_epochs = 2000,batch_size = 1000,train_thresh = 2,epoch_burn_in = 30,extra_epoch = 2000 ,n_weight_burn = 30
    #v6, final line: Epoch 2000. Train: 0.40001. Pure Train: 0.38750. Val: 0.38886. Sparse: 0.00645. Sparse Weight: 1.00000 Weight: 0.00606
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/raman_sim_layer_2_1.ckpt.meta")
    with tf.Session() as sess2:
        saver.restore(sess2,"models/raman_sim_layer_2_1.ckpt")
        params2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_2 = tuple(sess2.run(params2))
    tf.reset_default_graph()
        
#%% Evaluate second autoencoder


enc_2_train = sigmoid(np.matmul(a=enc_train,b = param_2[0]) + param_2[1])
enc_2_valid = sigmoid(np.matmul(a=enc_valid,b = param_2[0]) + param_2[1])
enc_2_test = sigmoid(np.matmul(a=enc_test,b = param_2[0]) + param_2[1])

#%% Plot training

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

#plt.savefig('Training_error_raman_sim_layer_2.png')





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

#plt.savefig('raman_sim_layer_2_latent_space_1006_epoch.png')


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

#plt.savefig('Statistics_activations_raman_sim_layer_2_706_epoch')
#plt.savefig('Statistics_activations_raman_sim_layer_2_1006_epoch')


#%%
plt.figure(figsize=(20,15))
count = 1
for c in included_classes:
    ax=plt.subplot(10,5,count)
    plt.boxplot(enc_2_valid[targets_valid==c,:])
    count+=1
#plt.savefig('raman_sim_boxplot_layer_2_1006_epoch.png')


#%%
y_train = np.eye(num_classes)[targets_train]
y_valid = np.eye(num_classes)[targets_valid]
y_test = np.eye(num_classes)[targets_test]
alpha = 0.003


#%%Train softmax layer
tf.reset_default_graph()
if True:
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
    saver.save(sess3, 'models/Softmax_raman_sim_1.ckpt')
    tf.reset_default_graph()
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/Softmax_raman_sim_1.ckpt.meta")
    with tf.Session() as sess3:
        saver.restore(sess3,"models/Softmax_raman_sim_1.ckpt")
        params3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_3 = tuple(sess3.run(params3))

#%% Plot training

updates = [i*x_train.shape[0] for i in range(1,np.shape(train_loss3)[0]+1)]
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.semilogy(updates,train_loss3,color="black")
plt.semilogy(updates,valid_loss3,color="gray")
plt.legend(['Train Error', 'Valid Error'])
plt.subplot(1,2,2)
plt.semilogy(updates,train_acc3,color="black")
plt.semilogy(updates,valid_acc3,color="gray")
plt.legend(["Train accuracy","valid accuracy"])

#plt.savefig('Training_error_raman_sim_softmax_layer.png')

#%% Fine tune
# only do 100 epoch
tf.reset_default_graph()
if True:
    out4 = AE.Sparse_Non_Neg_Stacked(x_train = x_train, x_valid = x_valid, 
#    out4 = Sparse_Non_Neg_Stacked(x_train = x_train, x_valid = x_valid, 
                              y_train = y_train, y_valid = y_valid, 
                              params = [param_1,param_2,param_3], num_epochs = 2000,
                              p_target = [0.05,1/hiddenSizeL2])
    
    sess4, train_loss4, train_acc4, valid_loss4, valid_acc4, train_reg4, train_sparse4 = tuple(out4)
    params4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_4 = tuple(sess4.run(params4))
    saver = tf.train.Saver()
    saver.save(sess4, 'models/raman_sim_finetune_v2.ckpt')
    tf.reset_default_graph()
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/raman_sim_finetune_v2.ckpt.meta")
    with tf.Session() as sess4:
        saver.restore(sess4,"models/raman_sim_finetune_v2.ckpt")
        params4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        param_4 = tuple(sess4.run(params4))
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

print(np.mean(np.equal(preds_train,targets_train)))
print(np.mean(np.equal(preds_valid,targets_valid)))
print(np.mean(np.equal(preds_test,targets_test)))
#%% 
updates = [i*x_train.shape[0] for i in range(1,np.shape(train_loss4)[0]+1)]
plt.figure(figsize=(12,6))
plt.subplot(3,1,1)
plt.semilogy(updates,train_loss4,color="black")
plt.semilogy(updates,valid_loss4,color="gray")
plt.legend(['Train Error', 'Valid Error'])

plt.subplot(3,1,2)
plt.plot(updates,train_acc4,color="black")
plt.plot(updates,valid_acc4,color="gray")
plt.legend(['Train Acc', 'Valid Acc'])

plt.subplot(3,1,3)
plt.semilogy(updates,train_reg4,color="gray")
plt.semilogy(updates,train_sparse4,color="black")
plt.legend(["Weight decay","Sparseness"])
#plt.savefig('Raman_sim_training_error_finetune.png')

#%% Check latent spaces - tsne
tsne = TSNE(n_components=2)
hidden_1_finetune = tsne.fit_transform(enc_fine_valid)
hidden_2_finetune = tsne.fit_transform(enc_2_fine_valid)

plt.figure(figsize=(15,7.5))
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
#plt.legend(legend_handles, classes_plot)

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
#plt.legend(legend_handles, classes_plot)
#plt.savefig("raman_sim_latent_space_finetune.png")

#%%
plt.figure(figsize=(20,15))
count = 1
for c in included_classes:
    ax=plt.subplot(10,5,count)
    plt.boxplot(enc_fine_valid[targets_valid==c,:])
    count+=1
#plt.savefig('raman_sim_finetune_boxplot_layer_1_2000_epoch.png')

#%%
plt.figure(figsize=(20,15))
count = 1
for c in included_classes:
    ax=plt.subplot(10,5,count)
    plt.boxplot(enc_2_fine_valid[targets_valid==c,:])
    count+=1
#plt.savefig('raman_sim_finetune_boxplot_layer_2_2000_epoch.png')

#%%
plt.figure(figsize=(12, 12))
counter = 0
for i in range(hiddenSizeL2):
    for j in range(hiddenSizeL2):
        counter +=1
        plt.subplot(hiddenSizeL2,hiddenSizeL2,counter)
        if i == j:
            if i == (hiddenSizeL2-1) & j == (hiddenSizeL2-1):
                plt.legend(legend_handles, classes_plot,ncol = 2)
        else:
            plt.cla()
#            plt.title('First layer Latent space')
            plt.xlabel('z'+str(j+1)), plt.ylabel('z'+str(i+1))
            color = iter(plt.get_cmap('brg')(np.linspace(0, 1.0, num_classes)))
            legend_handles = []
            for l, c in enumerate(included_classes):
                clr = next(color)
                h = plt.scatter(enc_2_fine_valid[targets_valid==c, j], enc_2_fine_valid[targets_valid==c, i], c=clr, s=5., lw=0, marker='o', )
                legend_handles.append(h)
            plt.grid('on')
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

#plt.savefig("Statistics_activations_raman_sim_finetune_layer_1_2000_epoch.png")

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
#plt.savefig("Statistics_activations_raman_sim_finetune_layer_2_2000_epoch.png")

#%% First encoding layer
encode_weights_final = param_4[0]
encode_weights_t = encode_weights_final.transpose()
plt.figure(figsize=(12, 12))
plt.cla()
plt.axis('off')
counter = 0
for i in range(10):
    plt.subplot(10,1,i+1)
    for j in range(5):
        plt.plot(encode_weights_final[:,counter])
        counter += 1
        
#plt.savefig('Encoding_weights_raman_sim_Fine_tune_layer_1.png')

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
    if False:
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

#plt.savefig("Raman_sim_confusion_matrix_fine_tune.png")
#%%
plt.plot(np.diag(cnf_valid))
plt.plot(np.diag(cnf_test))
#%%
plt.figure(figsize = (12,6))
plt.scatter(targets_valid,preds_valid)
plt.xlabel("Target")
plt.ylabel("Prediction")

#%%
#enc_fine_class0 = sigmoid(np.matmul(a=Data[:900,],b = param_4[0]) + param_4[1])
#enc_fine_class0 = sigmoid(np.matmul(a=Data[900:1800,],b = param_4[0]) + param_4[1])
enc_fine_class0 = sigmoid(np.matmul(a=Data[1800:2700,],b = param_4[0]) + param_4[1])

enc_2_fine_class0 = sigmoid(np.matmul(a=enc_fine_class0,b = param_4[2]) + param_4[3])

#%%
tmp = [i for i in range(50)]
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
for i in range(900):
    plt.scatter(tmp,enc_fine_class0[i,:])
tmp2 = [i for i in range(6)]
plt.subplot(2,1,2)
for i in range(900):
    plt.scatter(tmp2,enc_2_fine_class0[i,:])

#%%
#plt.plot(encode_weights_final[:,[4,30]])
plt.plot(encode_weights_final[:,[25,37]])