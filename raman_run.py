# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:35:54 2017

@author: jasla
"""

from __future__ import division, print_function
import os
path = "C:\\Users\\jasla\\Dropbox\\phd\\Kurser\\02456 Deep Learning\\Project\\python"

os.chdir(path)

Train_model = True
#Train_model = False
hiddenSizeL1 = 4;
hiddenSizeL2 = 3;
alpha = 0.1
tol = 10**(-4)


#%% load libraries

import AE_fun as AE
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','nbagg')
get_ipython().run_line_magic('matplotlib','inline')
import tensorflow as tf
import csv
from sklearn.metrics import confusion_matrix
import itertools
from PIL import Image

#%% Define some usefull functions
def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    return sig

def softmax(x):
    expx = np.exp(x.T - np.max(x,1)).T # avoid numerically instability
    soft_max = (expx.T * (1/ np.sum(expx,1))).T
    return(soft_max)
    
def running_average(dat,N):
    cumsum, running_aves = [0], []
    
    for i, x in enumerate(dat, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            running_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            running_aves.append(running_ave)
    return running_aves

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          include_colorbar = True):
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
    plt.title(title,fontsize = 20)
    if include_colorbar:
        plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45,fontsize = 15)
    plt.yticks(tick_marks, classes,fontsize = 15)
    if True:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",fontsize = 20)

    plt.tight_layout()
    plt.ylabel('True label',fontsize =20)
    plt.xlabel('Predicted label',fontsize =20)
#%% Load data    
n_samples = 100**2
nw = 1000
X1, X2, Xmix = AE.load_raman_map(filename = 'data/raman_100x100_wavenumbers1000_hotspots10_10dB_withinteractions.csv',
                                 do_plot=False,nw = nw,n_samples = n_samples)
X = np.concatenate((X1,X2),axis= 1).T

y1 = [0]*n_samples
y2 = [1]*n_samples
Y = np.concatenate((y1,y2))
included_classes = [0,1,2]
num_classes = len(included_classes)
classes_plot = included_classes
Xnmf = np.concatenate((X1,X2,Xmix),axis=1).T

with open("data/D-vec_100x100_1000.csv","r") as f:
    fileReader = csv.reader(f, delimiter = ",")
    D = []
    for line in fileReader:
        values = [float(val) for val in line]
        D += values
    D = np.array(D)

Dsmall = np.reshape(D[:10000],(100,100))
tmp = Image.fromarray(Dsmall)
#tmp = tmp.resize((50,50), Image.ANTIALIAS)
tmp = tmp.resize((25,100), Image.ANTIALIAS)
Dsmall = np.array(tmp).flatten()

dscaleSmall = (Dsmall - np.min(Dsmall))
dscaleSmall = dscaleSmall / np.max(dscaleSmall)
cols_small = []
for i in range(dscaleSmall.shape[0]):
    cols_small.append([.8*dscaleSmall[i]]*3)
cols_small = np.array(cols_small)

idx = D < tol
idx = np.concatenate((idx,idx))
Y[idx] = 2

D = np.concatenate((D,D,D))

dscale = (D - np.min(D))
dscale = dscale / np.max(dscale)
cols = []
for i in range(dscale.shape[0]):
    cols.append(np.array([.8,.8,.8])*dscale[i,])
    
#%% Plot raman mask
plt.figure(figsize=(8,8))
plt.imshow(np.reshape(D[:10000],(100,100)),cmap = "gray")
cb = plt.colorbar(fraction = .045)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=20)
plt.axis('off')

#%% Set up trainig, validation and test data
np.random.seed(seed = 1010)
n_train = int(2*n_samples * .5)
n_valid = int(2*n_samples * .3)
n_test = int(2*n_samples * .2)

idx = [i for i in range(X.shape[0])]
idx = np.random.permutation(idx)
idx_train = idx[:n_train]
idx_valid = idx[n_train:(n_train+n_valid)]
idx_test = idx[(n_train+n_valid):(n_train+n_valid+n_test)]

x_train = X[idx_train,]
x_valid = X[idx_valid,]
x_test = X[idx_test,]

D_train = D[idx_train]
D_valid = D[idx_valid]
D_test = D[idx_test]
cols_train = [cols[i] for i in idx_train]
cols_valid = [cols[i] for i in idx_valid]
cols_test = [cols[i] for i in idx_test]

targets_train = Y[idx_train]
targets_valid = Y[idx_valid]
targets_test = Y[idx_test]

print("training set dim(%i, %i)." % x_train.shape)
print("validation set dim(%i, %i)." % x_valid.shape)
print("test set dim(%i, %i)." % x_test.shape)

#%% Train first Auto Encoder
tf.reset_default_graph()
if Train_model:
    out = AE.Sparse_Non_Neg_AE(x_train = x_train, x_valid = x_valid,use_LS = True,num_epochs = 5000
                               , batch_size = 1000
                               , num_hidden = hiddenSizeL1
                               ,train_thresh = 10
                               , n_weight_burn = 10
                               , epoch_burn_in = 20
                               , extra_epoch = 200
                               , p_target = 1/hiddenSizeL1
                               , alpha = 0.1
                               )
    
    sess,train_loss,train_loss_pure,valid_loss,train_reg,train_sparse = tuple(out)
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_1 = tuple(sess.run(params))
    saver = tf.train.Saver()
    saver.save(sess, 'models/raman_sim_layer_1_.ckpt')
    tf.reset_default_graph()
else:
    saver = tf.train.import_meta_graph("models/raman_sim_layer_1_.ckpt.meta")
    sess = tf.Session()
    saver.restore(sess,"models/raman_sim_layer_1_.ckpt")
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_1 = tuple(sess.run(params))
    hiddenSizeL1 = param_1[1].shape[0]
    tf.reset_default_graph()

#%% Evaluate first Auto Encoder
fetches = ['l_enc/Sigmoid:0']
enc_train = sess.run(fetches, {'x_pl:0':x_train})
enc_valid = sess.run(fetches, {'x_pl:0':x_valid})
enc_test = sess.run(fetches, {'x_pl:0':x_test})

#%%Train second Auto Encoder
tf.reset_default_graph()
if Train_model:
    out2 = AE.Sparse_Non_Neg_AE(x_train = enc_train[0], x_valid = enc_valid[0],use_LS = True,
                                num_epochs = 2000,num_hidden = hiddenSizeL2
                                ,batch_size = 500
                                ,p_target = 1/hiddenSizeL2
                                ,train_thresh = 2
                                ,epoch_burn_in = 30
                                ,extra_epoch = 100
                                ,n_weight_burn = 30
                                ,alpha = 0.1
                                )
    
    sess2,train_loss2,train_loss_pure2,valid_loss2,train_reg2,train_sparse2 = tuple(out2)
    params2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_2 = tuple(sess2.run(params2))
    saver = tf.train.Saver()
    saver.save(sess2, 'models/raman_sim_layer_2_.ckpt')
    
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/raman_sim_layer_2_.ckpt.meta")
    sess2 = tf.Session()
    saver.restore(sess2,"models/raman_sim_layer_2_.ckpt")
    params2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_2 = tuple(sess2.run(params2))
    hiddenSizeL2 = param_2[1].shape[0]
    tf.reset_default_graph()
    
#%% Evaluate second Auto Encoder
fetches = ['l_enc/Sigmoid:0']
enc_2_train = sess2.run(fetches, {'x_pl:0':enc_train[0]})
enc_2_valid = sess2.run(fetches, {'x_pl:0':enc_valid[0]})
enc_2_test = sess2.run(fetches, {'x_pl:0':enc_test[0]})

#%%
y_train = np.eye(num_classes)[targets_train]
y_valid = np.eye(num_classes)[targets_valid]
y_test = np.eye(num_classes)[targets_test]

#%%
idx1 = np.where(targets_train == 0)[0]
idx2 = np.where(targets_train == 1)[0]
idx3 = np.where(targets_train == 2)[0]

plt.figure(figsize=(12,8))
ax = plt.subplot(3,1,1)
#plt.title("Class 1",fontsize = 20)
plt.ylabel("Class 1",fontsize = 15)
for i in range(1000):
    ax.plot(x_train[idx1[i],:], color = cols_train[idx1[i]],alpha = 0.2)
    
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

ax = plt.subplot(3,1,2)
#plt.title("Class 2",fontsize = 20)
plt.ylabel("Class 2",fontsize = 15)
for i in range(1000):
    ax.plot(x_train[idx2[i],:], color = cols_train[idx2[i]],alpha = 0.2)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

ax = plt.subplot(3,1,3)
#plt.title("Class 3",fontsize = 20)
plt.ylabel("Class 3",fontsize = 15)
for i in range(1000):
    ax.plot(x_train[idx3[i],:], color = cols_train[idx3[i]],alpha = 0.2)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
#%%Train softmax layer
tf.reset_default_graph()
if Train_model:
    out3 = AE.Non_neg_softmax(x_train = enc_2_train[0],y_train = y_train, x_valid = enc_2_valid[0],y_valid = y_valid,
                                num_epochs = 400
                                ,batch_size = 300
                                , alpha = 0.1
                                )
    
    sess3,train_loss3,train_acc3,valid_loss3,valid_acc3 = tuple(out3)
    params3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_3 = tuple(sess3.run(params3))
    saver = tf.train.Saver()
    saver.save(sess3, 'models/raman_sim_Softmax.ckpt')
    tf.reset_default_graph()
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/raman_sim_Softmax.ckpt.meta")
    sess3 = tf.Session()

    saver.restore(sess3,"models/raman_sim_Softmax.ckpt")
    params3 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_3 = tuple(sess3.run(params3))

#%% Fine tune
tf.reset_default_graph()
if Train_model:
    out4 = AE.Sparse_Non_Neg_Stacked(x_train = x_train, x_valid = x_valid, 
                              y_train = y_train, y_valid = y_valid, 
                              params = [param_1,param_2,param_3], num_epochs = 2000,
                              p_target = [1/hiddenSizeL1,1/hiddenSizeL2],
                              alpha = 0.1
                              )
    
    sess4, train_loss4, train_acc4, valid_loss4, valid_acc4, train_reg4, train_sparse4 = tuple(out4)
    params4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_4 = tuple(sess4.run(params4))
    saver = tf.train.Saver()
    saver.save(sess4, 'models/raman_sim_finetune_.ckpt')
    tf.reset_default_graph()
else:
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph("models/raman_sim_finetune_.ckpt.meta")
    sess4 = tf.Session()

    saver.restore(sess4,"models/raman_sim_finetune_.ckpt")
    params4 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    param_4 = tuple(sess4.run(params4))
        
#%% Evaluate final classifier
fetches = ['layer_0/Sigmoid:0','layer_1/Sigmoid:0','layer_2/Softmax:0']
enc_fine_train, enc_2_fine_train, yhat_train = sess4.run(fetches, {'x_pl:0':x_train})
enc_fine_valid, enc_2_fine_valid, yhat_valid = sess4.run(fetches, {'x_pl:0':x_valid})
enc_fine_test, enc_2_fine_test, yhat_test = sess4.run(fetches, {'x_pl:0':x_test})
enc_fine_mix, enc_2_fine_mix, yhat_mix = sess4.run(fetches, {'x_pl:0':Xmix.T})

preds_train = np.argmax(yhat_train,axis=1)
preds_valid = np.argmax(yhat_valid,axis=1)
preds_test = np.argmax(yhat_test,axis=1)

print(np.mean(np.equal(preds_train,targets_train)))
print(np.mean(np.equal(preds_valid,targets_valid)))
print(np.mean(np.equal(preds_test,targets_test)))

#%% First encoding layer
encode_weights_final = param_4[0]
encode_weights_t = encode_weights_final.transpose()
plt.figure(figsize=(8, 4))
plt.cla()
counter = 0
for i in range(hiddenSizeL1):
    for j in range(1):
        plt.plot(encode_weights_final[:,counter])
        counter += 1
        
plt.legend(["Component " + str(i) for i in range(hiddenSizeL1)])
plt.xlabel("Wavenumber")

#%% Second and Softmax layer
xticks = ["Weight " + str(i) for i in range(hiddenSizeL1)]
xticks += ["Bias"]
yticks = ["Out " + str(i) for i in range(hiddenSizeL2)]
canvas = np.zeros(shape = (hiddenSizeL1+1,hiddenSizeL2))
canvas[:-1,:] = param_4[2]
canvas[-1,:] = param_4[3]
canvas_soft = np.zeros(shape = (hiddenSizeL2+1,num_classes))
canvas_soft[:-1,:] = param_4[4]
canvas_soft[-1,:] = param_4[5]
vals = np.concatenate((canvas.flatten(),canvas_soft.flatten()),axis = 0)
vals.sort()

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Second layer",fontsize = 20)
ax = plt.imshow(canvas.T,cmap = "gray",aspect = "auto",vmin = min(vals),vmax = max(vals))
plt.xticks(np.arange(hiddenSizeL1+1),xticks,fontsize = 13)
plt.yticks(np.arange(hiddenSizeL2)-.25,yticks,fontsize = 13,rotation = 90)

xticks_soft = ["Weight " + str(i) for i in range(hiddenSizeL2)]
xticks_soft += ["Bias"]
yticks_soft = ["Class " + str(i) for i in range(num_classes)]

plt.subplot(1,2,2)
plt.title("Softmax layer",fontsize = 20)
ax = plt.imshow(canvas_soft.T,cmap = "gray",aspect = "auto",vmin = min(vals),vmax = max(vals))
plt.xticks(np.arange(hiddenSizeL2+1),xticks_soft,fontsize = 13)
plt.yticks(np.arange(num_classes)-.25,yticks_soft,fontsize = 13,rotation = 90)
cb = plt.colorbar(mappable=ax,orientation = "vertical",aspect = 20)
cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=20)

#%% Confusion matrices
cnf_valid = confusion_matrix(targets_valid,preds_valid); cnf_valid = cnf_valid.astype('float') / cnf_valid.sum(axis=1)[:, np.newaxis]
cnf_test = confusion_matrix(targets_test,preds_test); cnf_test = cnf_test.astype('float') / cnf_test.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plot_confusion_matrix(cnf_valid, classes=classes_plot, normalize=True,
                      title='Validation data',include_colorbar=False)
plt.subplot(1,2,2)
plot_confusion_matrix(cnf_test, classes=classes_plot, normalize=True,
                      title='Test data',include_colorbar = False)
#%% Evaluate on different mixtures
meas = []
with open("data/raman_50x50_wavenumbers1000_testset.csv","r") as f:
    fileReader = csv.reader(f, delimiter = ",")

    for line in fileReader:
        values = [float(val) for val in line]
#        meas = np.concatenate((meas,[values]))
        meas += [values]

substances = np.zeros((50*50,nw,8))
for i in range(len(meas)):
    tmp = np.reshape(meas[i],(nw,int(len(meas[i])/nw)))
    substances[:,:,i] = tmp.T

C = []
with open("data/raman_50x50_wavenumbers1000_testset_labels.csv","r") as f:
    fileReader = csv.reader(f, delimiter = ",")

    for line in fileReader:
#        break
        C += [float(i) for i in line]
        
C = np.reshape(C,(8,2))
Cpred = np.zeros((50*50,num_classes,8))
Encpred = np.zeros((50*50,num_classes,8))
fetches = ['layer_1/Sigmoid:0','layer_2/Softmax:0']
for i in range(8):
    enc,yhat = sess4.run(fetches, {'x_pl:0':substances[:,:,i]})
    Cpred[:,:,i] = yhat
    Encpred[:,:,i] = enc


#%% Plot probabilities as function of mask
Isort = Dsmall.argsort()
I2 = C[:,0].argsort()
n = 100
for i in range(8):
    plt.figure(figsize=(12,12))
    
    plt.plot(Dsmall,Cpred[:,0,I2[i]],'.',linewidth = 1,color = "#5e5eec")
    plt.plot(Dsmall,Cpred[:,1,I2[i]],'.',linewidth = 1,color = "#5ea55e")
    plt.plot(Dsmall,Cpred[:,2,I2[i]],'.',linewidth = 1,color = "#a55ea5")
             
    tmp = running_average(Cpred[Isort,0,I2[i]],N = n) 
    plt.semilogx(Dsmall[Isort][n-1:],tmp,linewidth = 3,color = "#0000ff")
    
    tmp = running_average(Cpred[Isort,1,I2[i]],N = n) 
    plt.semilogx(Dsmall[Isort][n-1:],tmp,linewidth = 3,color = "#008000")
    
    tmp = running_average(Cpred[Isort,2,I2[i]],N = n) 
    plt.semilogx(Dsmall[Isort][n-1:],tmp,linewidth = 3,color = "#800080")
    plt.title("Concentration 1: " +str(C[I2[i],0]),fontsize = 20)
    plt.ylabel("Probability",fontsize = 20)
    plt.xlabel("D",fontsize = 20)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.legend(["Class " + str(j+1) for j in range(3)],loc = "center left",fontsize = 15)
    plt.xlim((10**(-3),1))
    
#%% Plot predicted constrations using probabilities
CpredMean = np.mean(Cpred[:,:2,:],axis = 0)
CpredMean = CpredMean/np.sum(CpredMean,axis=0)
CpredWeight = np.matmul(a = Cpred.transpose(1,2,0),b = Dsmall.reshape((2500,1)))/np.sum(Dsmall)
plt.figure(figsize = (8,8))
plt.scatter(C[:,0],CpredMean[0,])
plt.scatter(C[:,0],CpredMean[1,])
plt.scatter(C[:,0],CpredWeight[0,:,0])
plt.scatter(C[:,0],CpredWeight[1,:,0])
plt.legend(["Sub. 1 - mean of prob","Sub. 2 - mean of prob",
            "Sub. 1 - weighted prob","Sub. 2 - weighted prob"],loc = "center left",fontsize = 15)
plt.plot([0,1],[0,1],color = "gray")
plt.plot([0,1],[1,0],color = "gray")
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Concentration of substance 1",fontsize = 20)
plt.title("Predicted concentration using probabilites",fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

#%% Plot predicted constrations using activations of last layer
EncMean = np.mean(Encpred[:,:,:],axis = 0)
EncMean = EncMean[[0,2],:]/np.sum(EncMean[[0,2],:],axis=0)
EncWeight = np.matmul(a = Encpred.transpose(1,2,0),b = Dsmall.reshape((2500,1)))/np.sum(Dsmall)
EncWeight = EncWeight[[0,2],:,:]/np.sum(EncWeight[[0,2],:,:],axis=0)

plt.figure(figsize = (8,8))
plt.scatter(C[:,0],EncMean[0,])
plt.scatter(C[:,0],EncMean[1,])
plt.scatter(C[:,0],EncWeight[0,:,0])
plt.scatter(C[:,0],EncWeight[1,:,0])
plt.legend(["Sub. 1 - mean of sigmoids","Sub. 2 - mean of sigmoids",
            "Sub. 1 - weighted sigmoids","Sub. 2 - weighted sigmoids"],loc = "upper center",fontsize = 15)
plt.plot([0,1],[0,1],color = "gray")
plt.plot([0,1],[1,0],color = "gray")
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel("Concentration of substance 1",fontsize = 20)

plt.title("Predicted concentration using sigmoids",fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

#%% Image of probabilities
I = C[:,0].argsort()
canvas = np.zeros((3*100,8*25))
for j in range(3):
    for i in range(8):
        canvas[j*100:(j+1)*100,i*25:(i+1)*25] = Cpred[:,j,I[i]].reshape((100,25))

plt.figure(figsize=(12,12))
plt.imshow(canvas,cmap="gray",aspect = "auto")
plt.xticks(np.arange(8)*25+12.5,[str(C[I[i],0]) for i in range(8)],fontsize = 20)
plt.yticks(np.arange(3)*100+50,["Class "+str(i+1) for i in range(3)],rotation = 90,fontsize = 20)


#%% Image of activations
I = C[:,0].argsort()
canvas = np.zeros((3*100,8*25))
for j in range(3):
    for i in range(8):
        canvas[j*100:(j+1)*100,i*25:(i+1)*25] = Encpred[:,j,I[i]].reshape((100,25))

plt.figure(figsize=(12,12))
plt.imshow(canvas,cmap="gray",aspect = "auto")
plt.xticks(np.arange(8)*25+12.5,[str(C[I[i],0]) for i in range(8)],fontsize = 20)
plt.yticks(np.arange(3)*100+50,["Class "+str(i+1) for i in range(3)],rotation = 90,fontsize = 20)