import os
path = "C:\\Users\\jasla\\Dropbox\\phd\\Kurser\\02456 Deep Learning\\Project\\python"

os.chdir(path)

# In[1]:

from __future__ import division, print_function
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','nbagg')
get_ipython().run_line_magic('matplotlib','inline')
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import tensorflow as tf
from tensorflow.python.framework.ops import reset_default_graph
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# In[2]:

from sklearn.utils import shuffle

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
x_train = data['X_train'][idxs_train].astype('float32')

# Since this is unsupervised, the targets are only used for validation.
targets_train = data['y_train'][idxs_train].astype('int32')
x_train, targets_train = shuffle(x_train, targets_train, random_state=1234)


x_valid = data['X_valid'][idxs_valid].astype('float32')
targets_valid = data['y_valid'][idxs_valid].astype('int32')

x_test = data['X_test'][idxs_test].astype('float32')
targets_test = data['y_test'][idxs_test].astype('int32')

print("training set dim(%i, %i)." % x_train.shape)
print("validation set dim(%i, %i)." % x_valid.shape)
print("test set dim(%i, %i)." % x_test.shape)


# In[3]:

#plot a few MNIST examples
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


# In[19]:

from tensorflow import layers
from tensorflow.contrib.layers import fully_connected 
from tensorflow.python.ops.nn import relu, sigmoid, tanh


#%% Define network

#load_model = True
load_model = False

# define in/output size
num_features = x_train.shape[1]
num_hidden_enc = 196
num_hidden_dec = 128
# reset graph
tf.reset_default_graph()
if load_model:
    saver = tf.train.import_meta_graph('models/model_manual_grad_weight_burn_all_classes_600epoch.ckpt.meta')
#    sess = tf.Session()
    with tf.Session() as sess:
        saver.restore(sess, 'models/model_manual_grad_weight_burn_all_classes_600epoch.ckpt')
        print("Model restored.")
        test_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        eval_param = tuple(sess.run(test_param))
    tf.reset_default_graph()

saver = tf.train.Saver() # we use this later to save the model
# define the model
# Input placeholder
x_pl = tf.placeholder(tf.float32, [None, num_features], 'x_pl')

if not load_model:
# Encoder
    l_enc = layers.dense(inputs=x_pl, units=num_hidden_enc, activation=sigmoid, name='l_enc')

# The latent variable layer, where we can try out activation functions
#l_z = layers.dense(inputs=l_enc, units=2, activation=None, name='l_z') # None indicates a linear output.
#l_z = layers.dense(inputs=l_enc, units=2, activation=tanh, name='l_z') # None indicates a linear output.

# Decoder
#l_dec = layers.dense(inputs=l_z, units=num_hidden_dec, activation=relu, name='l_dec')


# Output layer, sigmoid due to bounded pixel values in range [0,1]
#l_out = layers.dense(inputs=l_dec, units=num_features, activation=sigmoid) # iid pixel intensities between 0 and 1.
    l_out = layers.dense(inputs=l_enc, units=num_features, activation=sigmoid, name='l_dec')
else:
    l_enc = layers.dense(inputs=x_pl, units=num_hidden_enc, activation=sigmoid,
                         kernel_initializer=tf.constant_initializer(eval_param[0],dtype=tf.float32),bias_initializer=tf.constant_initializer(eval_param[1], dtype=tf.float32),
                         name='l_enc')
    l_out = layers.dense(inputs=l_enc, units=num_features,
                         kernel_initializer=tf.constant_initializer(eval_param[2],dtype=tf.float32),bias_initializer=tf.constant_initializer(eval_param[3], dtype=tf.float32),
                         activation=sigmoid, name='l_dec')
# Following we define the TensorFlow functions for training and evaluation.

#%% Define loss function

# Calculate loss - TRY another error function
# Squared error
#loss_per_pixel = tf.square(tf.subtract(l_out, x_pl))
# Binary cross-entropy error
eps = 10**(-10)
loss_per_pixel = - x_pl * tf.log(l_out+eps) - (1 - x_pl) * tf.log(1 - l_out + eps)
#loss_per_pixel = 
loss_pure = tf.reduce_mean(loss_per_pixel, name="mean_error")

# If you want regularization
#reg_scale = 0.0005
reg_scale = 0.003
#reg_scale = 0.000001
beta = 3.
#regularize = tf.contrib.layers.l2_regularizer(reg_scale)
#regularize = tf.contrib.layers.apply_regularization(non_neg_squared_weights_regularize,reg_scale)
#regularize = tf.contrib.layers.apply_regularization(tf.nn.relu(-weights),reg_scale)

params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#reg_term = tf.reduce_sum([regularize(param) for param in params])
#reg_term = tf.reduce_sum([regularize(tf.nn.relu(-param)) for param in params if param.name.endswith('kernel:0')])

reg_term = tf.reduce_sum([tf.reduce_sum(tf.nn.relu(-param)**2) for param in params if param.name.endswith('kernel:0')])
#reg_term = tf.reduce_sum([regularize(tf.nn.relu(-param)) for param in params if param.name.endswith('l_enc/kernel:0')])

#loss += reg_scale/2*reg_term

#sparse_param = 0.2
sparse_param = 0.05

sparse_pl = tf.placeholder(tf.float32, [1, 1], 'sparse_pl')
#p_act_enc = 1/x_train.shape[0]*(tf.reduce_sum(l_enc,0)) # ?batch_size
p_act_enc = tf.reduce_mean(l_enc,0)
#p_act_dec = 1/x_train.shape[0]*tf.reduce_sum(l_dec,0)
reg_sparse = tf.reduce_sum(sparse_param * tf.log(sparse_param/p_act_enc) + (1-sparse_param)*tf.log((1-sparse_param)/(1-p_act_enc)))

loss = loss_pure + sparse_pl * beta * reg_sparse



# define our optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.25)
#optimizer = tf.train.MomentumOptimizer(learning_rate = 0.01,momentum = 0.05)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
#optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
##

gradients = tf.gradients(loss,params)
gradients[0] -= reg_scale * tf.nn.relu(-params[0])
gradients[2] -= reg_scale * tf.nn.relu(-params[2])
train_op = optimizer.apply_gradients(zip(gradients,params))
##

# Training operator for applying the loss gradients in backpropagation update
#train_op = optimizer.minimize(loss)




#%% test the forward pass

# 
_x_test = np.zeros(shape=(32, num_features))
# initialize the Session
#if True:
#    with tf.Session() as sess:
sess = tf.Session()
# test the forward pass
sess.run(tf.global_variables_initializer())
#else:
#    with tf.Session() as sess:
   
        
feed_dict = {x_pl: _x_test}
res_forward_pass = sess.run(fetches=[l_out], feed_dict=feed_dict)
print("l_out", res_forward_pass[0].shape)


#saver.restore(sess, "models/model_hingeloss_all_classes_1200epoch.ckpt")

# Make the prediction
#prediction=tf.argmax(y_output, axis=1)
#store_prediction=prediction.eval(feed_dict={x_pl: x_test}, session=sess)

# In the training loop we sample each batch and evaluate the error, latent space and reconstructions every epoch.

#%% Train
use_free_bit = False
use_weight_burn_in = True


#batch_size = 32
batch_size = 1000
num_epochs = 600
#num_epochs = 200
#num_epochs = 10
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size
num_samples_valid = x_valid.shape[0]
num_batches_valid = num_samples_valid // batch_size
#updates = []

train_loss = []
train_loss_pure = []
train_sparse = []
train_reg = []
valid_loss = []
sparse_weight = [[0]]
sparse_counter = 0

cur_loss = 0
do_plot = True
a_nan = False
if do_plot:
    plt.figure(figsize=(12, 24))

try:
    for epoch in range(num_epochs):
        #Forward->Backprob->Update params
        cur_loss = []
        cur_loss_pure = []
        cur_sparse = []
        cur_reg = []
        for i in range(num_batches_train):
            idxs = np.random.choice(range(x_train.shape[0]), size=(batch_size), replace=False)    
            x_batch = x_train[idxs]
            # setup what to fetch, notice l
#            fetches_train = [train_op, loss, l_out, l_z]
            fetches_train = [train_op, loss, l_out,loss_pure,reg_sparse,reg_term]
            feed_dict_train = {x_pl: x_batch, sparse_pl: sparse_weight}
            # do the complete backprob pass
            res_train = sess.run(fetches_train, feed_dict_train)
#            _, batch_loss, train_out, train_z = tuple(res_train)
            _, batch_loss, train_out,batch_loss_pure,batch_sparse,batch_reg = tuple(res_train)
            cur_loss += [batch_loss]
            cur_loss_pure += [batch_loss_pure]
            cur_sparse += [beta*batch_sparse]
            cur_reg += [reg_scale/2*batch_reg]
            if np.isnan(cur_loss).any():
                a_nan = True
                break
        if a_nan:
            break
        train_loss += [np.mean(cur_loss)]
        train_loss_pure += [np.mean(cur_loss_pure)]
        train_sparse += [np.mean(cur_sparse)]
        train_reg += [np.mean(cur_reg)]
        
  
#        updates += [batch_size*num_batches_train*(epoch+1)]
        
        
        
        # evaluate
#        fetches_eval = [loss, l_out, l_z]
        fetches_eval = [loss, l_out,l_enc,reg_sparse,reg_term]
        feed_dict_eval = {x_pl: x_valid, sparse_pl: sparse_weight}                                
        res_valid = sess.run(fetches_eval, feed_dict_eval)
#        eval_loss, eval_out, eval_z = tuple(res_valid)
        eval_loss, eval_out, eval_enc,eval_sparse,eval_reg = tuple(res_valid)
        valid_loss += [eval_loss[0][0]]
        
        if (use_free_bit or use_weight_burn_in) and (sparse_weight[0][0] < 1):
            train_loss[-1] += (1-sparse_weight[0][0]) * train_sparse[-1]
            valid_loss[-1] += (1-sparse_weight[0][0]) * beta * eval_sparse
            
        
        if use_weight_burn_in and (sparse_weight[0][0] < 1) and (np.max(train_loss_pure[-5:]) < 0.20):
            sparse_weight[0][0] = np.logspace(-4,0,10)[sparse_counter]
            sparse_counter += 1
        

        if use_free_bit and (np.max(train_loss_pure[-5:]) < 0.20):
            sparse_weight = [[1]]
        

            
        print('Epoch %d. Train: %.5f. Pure Train: %.5f. Val: %.5f. Sparse: %.5f. Sparse Weight: %.5f Weight: %.5f'
              #%(epoch+1,train_loss[-1],valid_loss[-1],beta*eval_sparse,eval_reg))
              %(epoch+1,train_loss[-1],train_loss_pure[-1],valid_loss[-1],np.mean(cur_sparse),sparse_weight[0][0],np.mean(cur_reg)))
        if epoch == 0:
            continue
        
        # if you want to plot while training, uncomment the code below
        if do_plot and ((epoch+1) % 100 is 0 or (epoch+1) is num_epochs):
            updates = [i*batch_size*num_batches_train for i in range(1,np.shape(train_loss)[0]+1)]
            # Plotting
            plt.subplot(num_classes+1,2,1)
            plt.title('Error')
            plt.xlabel('Updates'), plt.ylabel('Error')
            plt.semilogy(updates, train_loss, color="black")
            plt.semilogy(updates, train_loss_pure, color="red")
            plt.semilogy(updates, valid_loss, color="grey")
            plt.legend(['Train Error','Pure error', 'Valid Error'])
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.grid('on')
            
            plt.subplot(num_classes+1,2,2)
            plt.title('Regularization')
            plt.xlabel('Updates'), plt.ylabel('Regularization')
            plt.semilogy(updates,train_sparse,color = "black")
            plt.semilogy(updates,train_reg,color = "gray")
            plt.legend(['Sparseness','Weight decay'])
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            plt.grid('on')
            
            c=0
            for k in range(3, 3 + num_classes*2, 2):
                plt.subplot(num_classes+1,2,k)
                plt.cla()
                plt.title('Inputs for %i' % included_classes[c])
                plt.axis('off')
                idx = 0
                canvas = np.zeros((28*10, 10*28))
                for i in range(10):
                    for j in range(10):
                        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_valid[targets_valid==included_classes[c]][idx].reshape((28, 28))
                        idx += 1
                plt.imshow(canvas, cmap='gray')
                
                plt.subplot(num_classes+1,2,k+1)
                plt.cla()
                plt.title('Reconstructions for %i' % included_classes[c])
                plt.axis('off')
                idx = 0
                canvas = np.zeros((28*10, 10*28))
                for i in range(10):
                    for j in range(10):
                        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = eval_out[targets_valid==included_classes[c]][idx].reshape((28, 28))
                        idx += 1
                plt.imshow(canvas, cmap='gray')
                c+=1
            plt.savefig("out51.png")
            display(Image(filename="out51.png"))
            clear_output(wait=True)
            
        
        
#    saver.save(sess, 'models/model_all_classes_1200epoch.ckpt')
#    saver.save(sess, 'models/model_hingeloss_all_classes_1200epoch.ckpt')
#    saver.save(sess, 'models/model_manual_grad_all_classes_600epoch.ckpt')
#    saver.save(sess, 'models/model_manual_grad_all_classes_600epoch_1000batch.ckpt')
#    saver.save(sess, 'models/model_manual_grad_all_classes_1200epoch.ckpt')
#    saver.save(sess, 'models/model_manual_grad_weight_burn_all_classes_600epoch.ckpt')
        
except KeyboardInterrupt:
    pass
    




# In[24]:

# Generate a subset of labeled data points

num_labeled = 10 # You decide on the size of the fraction...

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

idxs_train_l = []
for i in included_classes:
    idxs = np.where(targets_train == i)[0]
    idxs_train_l += np.random.choice(idxs, size=num_labeled).tolist()

x_train_l = x_train[idxs_train_l]
targets_train_l = targets_train[idxs_train_l]
print("labeled training set dim(%i, %i)." % x_train_l.shape)

plt.figure(figsize=(12, 7))
for i in range(num_classes*num_labeled):
    im = x_train_l[i].reshape((28, 28))
    plt.subplot(1, num_classes*num_labeled, i + 1)
    plt.imshow(im, cmap='gray')
    plt.axis('off')


# In[ ]:

#fetches_test = [loss, l_out, l_z,reg_sparse,l_enc]
fetches_test = [loss, l_out,reg_sparse,l_enc,reg_term,p_act_enc,params]
feed_dict_test = {x_pl: x_test,sparse_pl: [[1]]}
#feed_dict_test = {x_pl: x_valid}
res_test = sess.run(fetches_test, feed_dict_test)
#eval_loss, eval_out, eval_z,eval_sparse,eval_enc = tuple(res_test)
eval_loss, eval_out,eval_sparse,eval_enc, eval_reg,eval_p,eval_param = tuple(res_test)
#pca = PCA(n_components=2)
#pca.fit(eval_enc)
#eval_z = pca.transform(eval_enc)
tsne = TSNE(n_components=2)
eval_z = tsne.fit_transform(eval_enc)
 

#valid_loss += [eval_loss]
#%%
plt.figure(figsize=(12, 24))
plt.subplot(num_classes+1,3,1)
plt.title('Error')
plt.xlabel('Updates'), plt.ylabel('Error')
plt.semilogy(updates, train_loss, color="black")
plt.semilogy(updates, valid_loss, color="grey")
plt.semilogy(updates, train_loss_pure, color="red")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend(['Train Error', 'Valid Error','Pure Train Error'])
plt.grid('on')

plt.subplot(num_classes+1,3,2)
plt.title('Regularization')
plt.xlabel('Updates'), plt.ylabel('Regularization')
plt.semilogy(updates,train_sparse,color = "black")
plt.semilogy(updates,train_reg,color = "gray")
plt.legend(['Sparse reg','Weight decay'])
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')

plt.subplot(num_classes+1,3,3)
plt.cla()
plt.title('Latent space')
plt.xlabel('z0'), plt.ylabel('z1')
color = iter(plt.get_cmap('brg')(np.linspace(0, 1.0, num_classes)))
legend_handles = []
for i, c in enumerate(included_classes):
    clr = next(color)
    h = plt.scatter(eval_z[targets_test==c, 0], eval_z[targets_test==c, 1], c=clr, s=5., lw=0, marker='o', )
    legend_handles.append(h)
plt.grid('on')
plt.legend(legend_handles, included_classes)
        
c=0
for k in range(3, 3 + num_classes*2, 2):
    plt.subplot(num_classes+1,2,k)
    plt.cla()
    plt.title('Inputs for %i' % included_classes[c])
    plt.axis('off')
    idx = 0
    canvas = np.zeros((28*10, 10*28))
    for i in range(10):
        for j in range(10):
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_test[targets_test==included_classes[c]][idx].reshape((28, 28))
            idx += 1
    plt.imshow(canvas, cmap='gray')
        
    plt.subplot(num_classes+1,2,k+1)
    plt.cla()
    plt.title('Reconstructions for %i' % included_classes[c])
    plt.axis('off')
    idx = 0
    canvas = np.zeros((28*10, 10*28))
    for i in range(10):
        for j in range(10):
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = eval_out[targets_test==included_classes[c]][idx].reshape((28, 28))
            idx += 1
    plt.imshow(canvas, cmap='gray')
    c+=1
      
        
#plt.savefig("out_all_classes_1200_epoch.png")
#plt.savefig("out_hingeloss_all_classes_1200_epoch.png")
#plt.savefig("out_manual_grad_all_classes_600_epoch.png")
#plt.savefig("out_manual_grad_all_classes_1200_epoch.png")
#plt.savefig("out_manual_grad_all_weight_burn_classes_600_epoch.png")
#display(Image(filename="out52.png"))
clear_output(wait=True)


#%%
encode_weights = eval_param[0]
#encode_weights = eval_param[2].transpose()
encode_weights_t = encode_weights.transpose()
plt.figure(figsize=(12, 12))
plt.cla()
plt.axis('off')
canvas = np.zeros((28*14, 14*28))
idx = 0
for i in range(14):
    for j in range(14):
        tmp = encode_weights_t[idx]
#        idx_threshold = tmp > 1
#        tmp[idx_threshold] = 1
#        tmp = (tmp-min(tmp))/(max(tmp)-min(tmp))
        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = (tmp).reshape(28,28)
#        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = (tmp).reshape(28,28)*9.8+eval_param[3].reshape(28,28)
#        canvas[i*28:(i+1)*28, j*28:(j+1)*28] = np.log(tmp).reshape(28,28)
        idx += 1
        
#for i in range(28):
#    for j in range(28):
#        tmp = encode_weights[idx].reshape(14,14)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = (tmp)
#        canvas[i*14:(i+1)*14, j*14:(j+1)*14] = np.log(tmp)
#        idx += 1

cax = plt.imshow(canvas,cmap = 'gray')
cbar = plt.colorbar(cax)
#plt.savefig("Filters_man_grad_weight_burn_encode_all_600_epoch.png")
#plt.savefig("Filters_man_grad_weight_burn_decode_all_600_epoch.png")
#plt.savefig("Filters_man_grad_encode_all_1200_epoch.png")
#plt.savefig("Filters_log_man_grad_encode_all_1200_epoch.png")
#plt.savefig("Filters_man_grad_encode_all_600_epoch.png")
#plt.savefig("Filters_thresholded_man_grad_encode_all_600_epoch.png")
#plt.savefig("Filters_man_grad_decode_all_600_epoch.png")
#plt.savefig("Filters_thresholded_man_grad_decode_all_600_epoch.png")
#plt.savefig("Filters_man_grad_decode_all_1200_epoch.png")
#plt.savefig("Filters_log_man_grad_decode_all_1200_epoch.png")
#plt.savefig("Filters_log_hingeloss_decode_all_1200_epoch.png")
#plt.savefig("Filters_log_encode_hingeloss_all_1200_epoch.png")
#plt.savefig("Filters_log_hingeloss_decode_all_1200_epoch.png")
#%%
c = 0
plt.figure(figsize=(12, 24))
for k in range(1,3*num_classes+1,3):
    plt.subplot(num_classes,3,k)
    plt.cla()
    plt.title('Inputs for %i' % included_classes[c])
    plt.axis('off')
    idx = 0
    canvas = np.zeros((28*10, 10*28))
    for i in range(10):
        for j in range(10):
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_test[targets_test==included_classes[c]][idx].reshape((28, 28))
            idx += 1
    plt.imshow(canvas, cmap='gray')
    
    plt.subplot(num_classes,3,k+1)
    plt.cla()
    plt.title('Activation for %i' % included_classes[c])
    plt.axis('off')
    idx = 0
    canvas = np.zeros((14*10, 10*14))
    for i in range(10):
        for j in range(10):
            canvas[i*14:(i+1)*14, j*14:(j+1)*14] = eval_enc[targets_test==included_classes[c]][idx].reshape((14, 14))
            idx += 1
    cax = plt.imshow(canvas, cmap='gray',vmin=0,vmax=1)
    cbar = plt.colorbar(cax)
    plt.subplot(num_classes,3,k+2)
    plt.cla()
    plt.title('Reconstructions for %i' % included_classes[c])
    plt.axis('off')
    idx = 0
    canvas = np.zeros((28*10, 10*28))
    for i in range(10):
        for j in range(10):
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = eval_out[targets_test==included_classes[c]][idx].reshape((28, 28))
            idx += 1
    plt.imshow(canvas, cmap='gray')
    c+=1

#plt.savefig("Activations_man_grad_weight_burn_600_epoch.png")
#plt.savefig("Activations_all_classes_1200_epoch.png")
#plt.savefig("Activations_hingeloss_all_classes_1200_epoch.png")
#plt.savefig("Activations_man_grad_all_classes_600_epoch.png")
#plt.savefig("Activations_man_grad_all_classes_1200_epoch.png")

#%%

decode_weights = eval_param[2].transpose()
for i in range(196):
    print('Decode weights %.5f \t %.5f' %(min(decode_weights[i]),max(decode_weights[i])))
    
    

#%%
encode_weights = eval_param[0]
for i in range(784):
    print('Encode weights: %.5f \t %.5f' %(min(encode_weights[i]),max(encode_weights[i])))
#%%
def relu(x):
    out = [max(y,0) for y in x]
    return out

#%%
reg = 0
for i in range(784):
    reg += (sum([number**2 for number in relu(-encode_weights[i])]))

for i in range(196):
    reg += (sum([number**2 for number in relu(-decode_weights[i])]))
print(reg*reg_scale/2)

#%% Histogram
#plt.hist(np.concatenate(encode_weights))
#plt.hist(encode_weights)
#plt.hist(np.concatenate(decode_weights))
#plt.hist(decode_weights)

plt.hist((eval_enc))

#%%
plt.imshow(eval_param[3].reshape(28,28),cmap='gray')
#%%
plt.imshow(eval_param[1].reshape(14,14),cmap='gray')

#%%
plt.figure(figsize=(12, 6))
plt.subplot(2,2,1)

plt.xlabel('Hidden unit'), plt.ylabel('Max activation')
plt.plot(np.max(eval_enc,0), color="black")

plt.grid('on')
plt.subplot(2,2,2)
#plt.title('Error')
plt.xlabel('Test image'), plt.ylabel('Max activation')
plt.plot(np.max(eval_enc,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')
plt.subplot(2,2,3)
plt.xlabel('Hidden unit'), plt.ylabel('Mean activation')
plt.plot(np.mean(eval_enc,0), color="black")
plt.grid('on')
plt.subplot(2,2,4)
plt.xlabel('Test image'), plt.ylabel('Mean activation')
plt.plot(np.mean(eval_enc,1), color="black")
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.grid('on')

#plt.savefig("Statistics_activations_man_grad_weight_burn_600_epoch.png")