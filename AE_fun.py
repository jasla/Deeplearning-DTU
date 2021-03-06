# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:49:51 2017

@author: jasla
"""

#%% load required packages 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib','nbagg')
get_ipython().run_line_magic('matplotlib','inline')
import numpy as np
import tensorflow as tf
from tensorflow import layers
from tensorflow.python.ops.nn import relu, sigmoid
import matplotlib.pyplot as plt
import csv

#%%
def load_raman_map(filename=None,do_plot = False,nw = 500, n_samples = 1600):
    if not filename:
        filename = 'data/raman_40x40_wavenumbers500_hotspots10_0dB_withinteractions.csv'
    
    meas = np.zeros(shape = (0,nw*n_samples))
    with open(filename,"r") as f:
        fileReader = csv.reader(f, delimiter = ",")
    
        for line in fileReader:
            values = [float(val) for val in line]
            meas = np.concatenate((meas,[values]))
            
    print(meas.shape)
    substance1_map = meas[0,:]
    substance1_r = substance1_map.reshape(nw,n_samples)

    substance2_map = meas[1,:]
    substance2_r = substance2_map.reshape(nw,n_samples)

    mix_map = meas[2,:]
    mix_r = mix_map.reshape(nw,n_samples)
    
    
    if do_plot:
        plt.figure(figsize=(12,12))
        plt.subplot(3,1,1)
        plt.plot(substance1_r)
    
        plt.subplot(3,1,2)
        plt.plot(substance2_r)
        plt.subplot(3,1,3)
        plt.plot(mix_r)
    
        plt.show(block=False)

    return substance1_r,substance2_r,mix_r


#%%
def Sparse_Non_Neg_AE(x_train, x_valid, **kwargs):
    """
    Train a sparse non-negative auto encoder
    For details see     Ehsan Hosseini-Asl, Jacek M. Zurada, Life Fellow and Olfa Nasraoui
                        Deep Learning of Part-Based Representation of 
                        Data Using Sparse Autoencoders With 
                        Nonnegativity Constraints
    
    Input variables:
        Mandatory:
            x_train:                Matrix holding the training data with observations in the rows
            x_valid:                Matrix holding the validation data with observations in the rows
            
        Non-mandatory
            modelpath:              Path to model that needs refinement
            train_thresh:           Threshold defining when to start enforcing sparsity (default 0.0194 if least squares is used else 0.15)
            num_hidden:             Number of hidden units (default 196)
            alpha:                  Non-negativity constrain penalty (default 0.003)
            beta:                   Sparsity constrain penalty (default 3)
            tau:                    Learning rate (default 0.001)
            p_target:               Sparsity target
            use_LS:                 Use least squares reconstruction error (default False - i.e. use cross-entropy)
            use_weight_burn_in:     Use weight burn in on sparsity constraint (default True) sparsity is introduced on a log-scale when a reconstruction error of train_tresh has been reached. 
            n_weight_burn:          How many steps used to introduce sparsity consraint when using weight burn in (default 10)
            epoch_burn_in:          Number of epochs between increments of weight burn in (default 5)
            log_start:              Lower limit of logspace used for weight burn in (default -4)
            weights_burn_in:        Manually define the weights in weight burn in (default np.logspace(log_start,log_end,n_weight_burn))
            use_free_bit:           Introduce full sparsity when a reconstruction error of train_tresh is reached (default False)
            extra_epoch:            Number of epochs to perform after full sparsity constraint has been introduced (default 2)
            num_epochs:             Maximum number of epochs (default 1000)
            batch_size:             Batch size for training (default 1000)
            
    Output:
        sess:                       The session
        train_loss                  Training loss
        train_loss_pure             Reconstruction error
        valid_loss                  Validation loss
        train_reg                   Non-negativity constrain regularization for training
        train_sparse                Sparsity constrain regularization for training
    """
    # Unload parameters
    modelpath = kwargs.get('modelpath',None)
    train_thresh = kwargs.get('train_thresh',None)
    num_hidden = kwargs.get('num_hidden',196)
    alpha = kwargs.get('alpha',0.003)
    beta = kwargs.get('beta',3)
    tau = kwargs.get('tau',0.001)
    p_target = kwargs.get('p_target',0.05 )
    use_LS = kwargs.get('use_LS',False)
    use_weight_burn_in = kwargs.get('use_weight_burn_in',True)
    use_free_bit = kwargs.get('use_free_bit',False)
    num_epochs = kwargs.get('num_epochs',1000)
    batch_size = kwargs.get('batch_size',1000)
    n_weight_burn = kwargs.get('n_weight_burn',10)
    extra_epoch = kwargs.get('extra_epoch',2)
    epoch_burn_in = kwargs.get("epoch_burn_in",5)
    log_start = kwargs.get('log_start',-4)
    log_end = 0
    weights_burn_in = kwargs.get('weights_burn_in',np.logspace(log_start,log_end,n_weight_burn))
    sparse_weight_start = kwargs.get("sparse_weight_start",0)
    
    eps = 10**(-10)
    
    if not modelpath:
        load_model=False
    else:
        if isinstance(modelpath,str):
            load_model=True

    if not train_thresh:
        if use_LS:
            train_thresh = 0.0194
        else:
            train_thresh = 0.15

    # Define network
    # define in/output size
    num_features = x_train.shape[1]
    
    # Restore provided model
    tf.reset_default_graph()
    if load_model:
        saver = tf.train.import_meta_graph(modelpath + ".meta")
    
        with tf.Session() as sess:
            saver.restore(sess,modelpath)
            print("Model restored.")
            test_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            eval_param = tuple(sess.run(test_param))
            
        tf.reset_default_graph()
    
    
    # Define the model
    # Input placeholder
    x_pl = tf.placeholder(tf.float32, [None, num_features], 'x_pl')
    
    if not load_model:
        # Encoder
        l_enc = layers.dense(inputs=x_pl, units=num_hidden, activation=sigmoid, name='l_enc')
        l_out = layers.dense(inputs=l_enc, units=num_features, activation=sigmoid, name='l_dec')
    else:
        l_enc = layers.dense(inputs=x_pl, units=num_hidden, activation=sigmoid,
                             kernel_initializer=tf.constant_initializer(eval_param[0],dtype=tf.float32),
                             bias_initializer=tf.constant_initializer(eval_param[1], dtype=tf.float32),
                             name='l_enc')
        l_out = layers.dense(inputs=l_enc, units=num_features,
                             kernel_initializer=tf.constant_initializer(eval_param[2],dtype=tf.float32),
                             bias_initializer=tf.constant_initializer(eval_param[3], dtype=tf.float32),
                             activation=sigmoid, name='l_dec')

    # Define loss function
    if use_LS: # Squared error
        loss_per_pixel = tf.square(tf.subtract(l_out, x_pl)); 
    else: # Binary cross-entropy error
        loss_per_pixel = - x_pl * tf.log(l_out+eps) - (1 - x_pl) * tf.log(1 - l_out + eps);
    
    loss_per_image = tf.reduce_sum(loss_per_pixel,1)
    
    loss_pure = 0.5*tf.reduce_mean(loss_per_image, name="mean_error")
    
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    reg_term = tf.reduce_sum([tf.reduce_sum(tf.nn.relu(-param)**2) for param in params if param.name.endswith('kernel:0')])
    
    
    sparse_pl = tf.placeholder(tf.float32, [1, 1], 'sparse_pl')
    p_act_enc = tf.reduce_mean(l_enc,0)
    reg_sparse = tf.reduce_sum(p_target * tf.log(p_target/(p_act_enc+eps)) + (1-p_target)*tf.log((1-p_target)/(1-p_act_enc+eps)))
    

    loss = loss_pure + sparse_pl * beta * reg_sparse + alpha/2 * reg_term
    
    # define our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=tau)
    
    train_op = optimizer.minimize(loss)


    # Test the forward pass
    _x_test = np.zeros(shape=(32, num_features))
    
    # initialize the Session
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
            
    feed_dict = {x_pl: _x_test}
    res_forward_pass = sess.run(fetches=[l_out], feed_dict=feed_dict)
    print("l_out", res_forward_pass[0].shape)


    # Train
    full_weight_burn = False
    epoch_weight_burn = 0
    
    num_samples_train = x_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    
    train_loss = []
    train_loss_pure = []
    train_sparse = []
    train_reg = []
    valid_loss = []
    sparse_weight = [[sparse_weight_start]]
    sparse_counter = 0
    
    cur_loss = 0
    
    try:
        for epoch in range(num_epochs):
            cur_loss = []
            cur_loss_pure = []
            cur_sparse = []
            cur_reg = []
            for i in range(num_batches_train):
                idxs = np.random.choice(range(x_train.shape[0]), size=(batch_size), replace=False)    
                x_batch = x_train[idxs]
                
                fetches_train = [train_op, loss, l_out,loss_pure,reg_sparse,reg_term]
                feed_dict_train = {x_pl: x_batch, sparse_pl: sparse_weight}

                res_train = sess.run(fetches_train, feed_dict_train)
                _, batch_loss, train_out,batch_loss_pure,batch_sparse,batch_reg = tuple(res_train)
                cur_loss += [batch_loss]
                cur_loss_pure += [batch_loss_pure]
                cur_sparse += [beta*batch_sparse]
                cur_reg += [alpha/2*batch_reg]
            
            
            train_sparse += [np.mean(cur_sparse)]
            train_reg += [np.mean(cur_reg)]
            train_loss += [np.mean(cur_loss)]
            train_loss_pure += [np.mean(cur_loss_pure)]
            
            # evaluate
            fetches_eval = [loss, l_out,l_enc,reg_sparse,reg_term]
            feed_dict_eval = {x_pl: x_valid, sparse_pl: sparse_weight}                                
            res_valid = sess.run(fetches_eval, feed_dict_eval)
            eval_loss, eval_out, eval_enc,eval_sparse,eval_reg = tuple(res_valid)
            valid_loss += [eval_loss[0][0]]
            
            
            if (use_free_bit or use_weight_burn_in) and (sparse_weight[0][0] < 1):
                train_loss[-1] += (1-sparse_weight[0][0]) * train_sparse[-1]
                valid_loss[-1] += (1-sparse_weight[0][0]) * beta * eval_sparse
                
            print('Epoch %d. Train: %.5f. Pure Train: %.5f. Val: %.5f. Sparse: %.5f. Sparse Weight: %.5f Weight: %.5f'
                  %(epoch+1,train_loss[-1],train_loss_pure[-1],valid_loss[-1],np.mean(cur_sparse),sparse_weight[0][0],np.mean(cur_reg)))
            
            
            if use_weight_burn_in and epoch >= (epoch_weight_burn+epoch_burn_in) and (sparse_weight[0][0] < 1) and (np.max(train_loss_pure[-epoch_burn_in:]) < train_thresh):
                sparse_weight[0][0] = weights_burn_in[sparse_counter]
                sparse_counter += 1
                epoch_weight_burn = epoch
                if (sparse_counter) == n_weight_burn:
                    full_weight_burn = True
                    
            
            if full_weight_burn:
                if epoch == (epoch_weight_burn+extra_epoch):
                    break
    
            if use_free_bit and (np.max(train_loss_pure[-epoch_burn_in:]) < train_thresh):
                sparse_weight = [[1]]
                full_weight_burn = True
                epoch_weight_burn = epoch
            
            if epoch == 0:
                continue
            
    except KeyboardInterrupt:
        pass
    
    return(sess,train_loss,train_loss_pure,valid_loss,train_reg,train_sparse)
    
    

#%%
def Non_neg_softmax(x_train, x_valid,y_train,y_valid, **kwargs):          
    """
    Train a softmax layer on top of a deep non-negative constraint neural network
    
    Input variables:
        Mandatory:
            x_train:        Matrix holding the training data with observations in the rows
            x_valid:        Matrix holding the validation data with observations in the rows
            
        Non-mandatory
            modelpath:      Path to model that needs refinement
            alpha:          Non-negativity constrain penalty (default 0.003)
            tau:            Learning rate (default 0.001)
            num_epochs:     Maximum number of epochs (default 1000)
            batch_size:     Batch size for training (default 1000)
            
    Output:
        sess:               The session
        train_loss          Training loss
        train_acc           Training accuracy
        valid_loss          Validation loss
        valid_acc           Validation accuracy
    """

    # Unload check
    modelpath = kwargs.get('modelpath',None)
    alpha = kwargs.get('alpha',0.003)
    tau = kwargs.get('tau',0.001)
    num_epochs = kwargs.get('num_epochs',1000)
    batch_size = kwargs.get('batch_size',1000)
    eps = 10**(-10)
    
    if not modelpath:
        load_model=False
    else:
        if isinstance(modelpath,str):
            load_model=True



    # Define network
    # define in/output size
    num_features = x_train.shape[1]
    num_classes = y_train.shape[1]
    
    # Restore provided model
    tf.reset_default_graph()
    if load_model:
        saver = tf.train.import_meta_graph(modelpath + ".meta")
    
        with tf.Session() as sess:
            saver.restore(sess,modelpath)
            print("Model restored.")
            test_param = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            eval_param = tuple(sess.run(test_param))
            
        tf.reset_default_graph()
    
    
    # define the model
    # Placeholders
    x_pl = tf.placeholder(tf.float32, [None, num_features], 'x_pl')
    y_pl = tf.placeholder(tf.float32, [None, num_classes], name='y_pl')
    
    if not load_model:
        # Encoder
        l_softmax = layers.dense(inputs=x_pl, units=num_classes, activation=tf.nn.softmax, name='l_softmax')
    else:
        l_softmax = layers.dense(inputs=x_pl, units=num_classes, activation=tf.nn.softmax,
                             kernel_initializer=tf.constant_initializer(eval_param[0],dtype=tf.float32),bias_initializer=tf.constant_initializer(eval_param[1], dtype=tf.float32),
                             name='l_softmax')
    
    y = l_softmax
    # Define loss function    
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+eps), reduction_indices=[1])

    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)
    
    loss_softmax = cross_entropy
    
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    reg_term = tf.reduce_sum([tf.reduce_sum(tf.nn.relu(-param)**2) for param in params if param.name.endswith('kernel:0')])
    
    loss = loss_softmax
    
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))

    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # define our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=tau)
    
    # Manually define gradients
    gradients = tf.gradients(loss,params)
    gradients[0] -= alpha * relu(-params[0])
    train_op = optimizer.apply_gradients(zip(gradients,params))
    
    if not load_model:
        saver = tf.train.Saver() # we use this later to save the model


    # test the forward pass
    _x_test = np.zeros(shape=(32, num_features))
    # initialize the Session
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
            
    feed_dict = {x_pl: _x_test}
    res_forward_pass = sess.run(fetches=[l_softmax], feed_dict=feed_dict)
    print("l_softmax", res_forward_pass[0].shape)


    # Train 
    num_samples_train = x_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    
    train_loss = []
    train_reg = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    
    try:
        for epoch in range(num_epochs):
            cur_loss = []
            cur_acc = []
            cur_reg = []
            
            for i in range(num_batches_train):
                idxs = np.random.choice(range(x_train.shape[0]), size=(batch_size), replace=False)    
                x_batch = x_train[idxs]
                y_batch = y_train[idxs]
                
                # setup what to fetch
                fetches_train = [train_op, loss,reg_term,accuracy]
                feed_dict_train = {x_pl: x_batch,
                                   y_pl: y_batch}
                
                # do the complete backprob pass
                res_train = sess.run(fetches_train, feed_dict_train)
                _, batch_loss, batch_reg, batch_acc = tuple(res_train)
                cur_loss += [batch_loss]
                cur_acc += [batch_acc]
                cur_reg += [alpha/2*batch_reg]
            
            
            train_reg += [np.mean(cur_reg)]
            train_loss += [np.mean(cur_loss)]
            train_loss[-1] += train_reg[-1]
            train_acc += [np.mean(cur_acc)]
            
            # evaluate
            fetches_eval = [loss, reg_term, accuracy]
            feed_dict_eval = {x_pl: x_valid, y_pl: y_valid}                                
            res_valid = sess.run(fetches_eval, feed_dict_eval)
            eval_loss, eval_reg, eval_acc = tuple(res_valid)
            valid_loss += [eval_loss]
            valid_acc += [eval_acc]
            

            print('Epoch %d. Train: %.5f. Train Acc: %.5f. Val: %.5f. Val Acc: %.5f. Weight: %.5f'
                  %(epoch+1,train_loss[-1],train_acc[-1],valid_loss[-1],np.mean(valid_acc[-1]),np.mean(cur_reg)))
            
            if epoch == 0:
                continue
            
    except KeyboardInterrupt:
        pass
    
    return(sess,train_loss,train_acc,valid_loss,valid_acc)
#%% Stacked Network
def make_hidden(input_pl, hidden_num,activation_fun,param,layer_name):
    """
    Helper function for creating the hidden layers in 
    """
    return layers.dense(inputs=input_pl, units=hidden_num, 
                       activation = activation_fun,
                             kernel_initializer=tf.constant_initializer(param[0],dtype=tf.float32),
                             bias_initializer=tf.constant_initializer(param[1], dtype=tf.float32),
                             name=layer_name)
  
def Sparse_Non_Neg_Stacked(x_train, x_valid,y_train,y_valid,params, **kwargs):
    """
    Fine tune a sparse non-negative deep neural network
    For details see     Ehsan Hosseini-Asl, Jacek M. Zurada, Life Fellow and Olfa Nasraoui
                        Deep Learning of Part-Based Representation of 
                        Data Using Sparse Autoencoders With 
                        Nonnegativity Constraints
    
    Input variables:
        Mandatory:
            x_train:        Matrix holding the training data with observations in the rows
            y_train:        Vector holding the training labels
            x_valid:        Matrix holding the validation data with observations in the rows
            y_valid:        Vector holding the validation labels
            params:         List holding tuples of parameters for each hidden layer. With params[i][0] being the weights of the i'th layer and params[i][1] being the biases
            
        Non-mandatory
            alpha:          Non-negativity constrain penalty (default 0.003)
            beta:           Sparsity constrain penalty (default 3)
            tau:            Learning rate (default 0.001)
            p_target:       Sparsity target
            num_epochs:     Maximum number of epochs (default 1000)
            batch_size:     Batch size for training (default 1000)
            names:          List holding the names for the hidden layes (default layer_0,...,layer_N for N layers)
            activations:    List holding the activation functions for the hidden layers (default tf.nn.sigmoid for all layers except the last which is tf.nn.softmax)
            
    Output:
        sess:                       The session
        train_loss                  Training loss
        train_acc                   Training accuracy
        valid_loss                  Validation loss
        valid_acc                   Validation accuracy
        train_reg                   Non-negativity constrain regularization for training
        train_sparse                Sparsity constrain regularization for training
    """
    alpha = kwargs.get('alpha',0.003)
    beta = kwargs.get("beta",3)
    p_target = kwargs.get("p_target",0.05)
    tau = kwargs.get('tau',0.001)
    num_epochs = kwargs.get('num_epochs',200)
    batch_size = kwargs.get('batch_size',1000)
    names = kwargs.get("names",None)
    activations = kwargs.get("activations",None)
    
    eps = 10**(-10)
    
    if not names:
        names = ["layer_" + str(i) for i in range(len(params))]
    
    if not activations:
        activations = [tf.nn.sigmoid for i in range(len(params)-1)]
        activations.append(tf.nn.softmax)
    
    # Define network
    # define in/output size
    num_features = x_train.shape[1]
    num_classes = y_train.shape[1]
    
    tf.reset_default_graph()
        
    x_pl = tf.placeholder(tf.float32,[None,x_train.shape[1]],'x_pl')
    y_pl = tf.placeholder(tf.float32, [None, num_classes], name='y_pl')
    
    n_units = [params[i][0].shape[1] for i in range(len(params))]
    
    nn_layers = []
    nn_layers.append(make_hidden(input_pl = x_pl,hidden_num = n_units[0],
                         activation_fun = activations[0], param = params[0], layer_name = names[0]))
    for i in range(1,len(params)):
        nn_layers.append(make_hidden(input_pl = nn_layers[i-1], hidden_num = n_units[i],
                                     activation_fun = activations[i], param = params[i], layer_name = names[i]))
    y = nn_layers[-1]
    print("layers generated")
    
    
    # Define cost function
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+eps), reduction_indices=[1])

    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)
    
    loss_softmax = cross_entropy
    
    parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    reg_term = alpha/2 * tf.reduce_sum([tf.reduce_sum(tf.nn.relu(-param)**2) for param in parameters if param.name.endswith('kernel:0')])
    
    p_act_enc = []
    for i in range(len(nn_layers)-1):
        p_act_enc += [tf.reduce_mean(nn_layers[i],0)]
    
    if len(p_target) == 1:
        p_target = [p_target for i in range(len(params))]
    
    reg_sparse = 0
    for i in range(len(p_act_enc)):
        reg_sparse += beta * tf.reduce_sum(p_target[i] * tf.log(p_target[i]/p_act_enc[i]) + (1-p_target[i])*tf.log((1-p_target[i])/(1-p_act_enc[i])))
    
    loss = loss_softmax + reg_sparse
    
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))

    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # define our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=tau)
    
    # Manually define gradients
    gradients = tf.gradients(loss,parameters)
    for i in range(len(parameters)):
        param = parameters[i]
        if param.name.endswith("kernel:0"):
            gradients[i] -= alpha * relu(-param)
            
    train_op = optimizer.apply_gradients(zip(gradients,parameters))
    

    # Test forward pass
    _x_test = np.zeros(shape=(32, num_features))
    # initialize the Session
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
            
    feed_dict = {x_pl: _x_test}
    res_forward_pass = sess.run(fetches=[nn_layers[-1]], feed_dict=feed_dict)
    print("l_softmax", res_forward_pass[0].shape)
    
    # Perform training operation
    num_samples_train = x_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    #updates = []
    
    train_loss = []
    train_reg = []
    train_acc = []
    train_sparse = []
    valid_loss = []
    valid_acc = []
    
    
    try:
        for epoch in range(num_epochs):
            cur_loss = []
            cur_acc = []
            cur_reg = []
            cur_sparse = []
            
            for i in range(num_batches_train):
                idxs = np.random.choice(range(x_train.shape[0]), size=(batch_size), replace=False)    
                x_batch = x_train[idxs]
                y_batch = y_train[idxs]
                
                # setup what to fetch
                fetches_train = [train_op, loss,accuracy,reg_term,reg_sparse]
                feed_dict_train = {x_pl: x_batch,
                                   y_pl: y_batch}
                
                # do the complete backprob pass
                res_train = sess.run(fetches_train, feed_dict_train)
                _, batch_loss, batch_acc, batch_reg, batch_sparse = tuple(res_train)
                cur_loss += [batch_loss]
                cur_acc += [batch_acc]
                cur_reg += [batch_reg]
                cur_sparse += [batch_sparse]
            
            train_reg += [np.mean(cur_reg)]
            train_sparse += [np.mean(cur_sparse)]
            train_loss += [np.mean(cur_loss)]
            train_loss[-1] += train_reg[-1]
            train_acc += [np.mean(cur_acc)]
            
            # evaluate
            fetches_eval = [loss, reg_term, accuracy]
            feed_dict_eval = {x_pl: x_valid, y_pl: y_valid}                                
            res_valid = sess.run(fetches_eval, feed_dict_eval)
            eval_loss, eval_reg, eval_acc = tuple(res_valid)
            valid_loss += [eval_loss]
            valid_acc += [eval_acc]
            

            print('Epoch %d. Train: %.5f. Train Acc: %.5f. Val: %.5f. Val Acc: %.5f. Weight: %.5f Sparse: %.5f'
                  %(epoch+1,train_loss[-1],train_acc[-1],valid_loss[-1],np.mean(valid_acc[-1]),train_reg[-1],train_sparse[-1]))
            
            if epoch == 0:
                continue
        
    except KeyboardInterrupt:
        pass
    
    return(sess,train_loss,train_acc,valid_loss,valid_acc,train_reg,train_sparse)