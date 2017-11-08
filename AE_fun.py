# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:49:51 2017

@author: jasla
"""

#%% load required packages 
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from sklearn.utils import shuffle
from tensorflow import layers
#from tensorflow.contrib.layers import fully_connected 
from tensorflow.python.ops.nn import relu, sigmoid, tanh

def Sparse_Non_Neg_AE(x_train, x_valid, **kwargs):          


#%% Unload check
    do_plot = kwargs.get('do_plot',False)
    plot_filename = kwargs.get('plot_filename','trainAE.png')
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
    eps = kwargs.get('eps',10**(-10))
    epoch_burn_in = kwargs.get("epoch_burn_in",5)
    log_start = kwargs.get('log_start',-4)
    log_end = kwargs.get('log_end',0)
    weights_burn_in = kwargs.get('weights_burn_in',np.logspace(log_start,log_end,n_weight_burn))
#    do_plot = kwargs.get('do_plot',False) 
    
#    print("n_weight_burn = %f" % n_weight_burn)
    
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

#    use_free_bit = not use_weight_burn_in

#%% Define network
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
    
    
    # define the model
    # Input placeholder
    x_pl = tf.placeholder(tf.float32, [None, num_features], 'x_pl')
    
    if not load_model:
        # Encoder
        l_enc = layers.dense(inputs=x_pl, units=num_hidden, activation=sigmoid, name='l_enc')
    
        # Output layer, sigmoid due to bounded pixel values in range [0,1]
        l_out = layers.dense(inputs=l_enc, units=num_features, activation=sigmoid, name='l_dec')
    else:
        l_enc = layers.dense(inputs=x_pl, units=num_hidden, activation=sigmoid,
                             kernel_initializer=tf.constant_initializer(eval_param[0],dtype=tf.float32),bias_initializer=tf.constant_initializer(eval_param[1], dtype=tf.float32),
                             name='l_enc')
        l_out = layers.dense(inputs=l_enc, units=num_features,
                             kernel_initializer=tf.constant_initializer(eval_param[2],dtype=tf.float32),bias_initializer=tf.constant_initializer(eval_param[3], dtype=tf.float32),
                             activation=sigmoid, name='l_dec')

#%% Define loss function
    if use_LS: # Squared error
        loss_per_pixel = tf.square(tf.subtract(l_out, x_pl)); 
    else: # Binary cross-entropy error
        loss_per_pixel = - x_pl * tf.log(l_out+eps) - (1 - x_pl) * tf.log(1 - l_out + eps);
    
    loss_per_image = tf.reduce_sum(loss_per_pixel,1)
    
    loss_pure = 0.5 * tf.reduce_mean(loss_per_image, name="mean_error")
    
    params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    
    reg_term = tf.reduce_sum([tf.reduce_sum(tf.nn.relu(-param)**2) for param in params if param.name.endswith('kernel:0')])
    
    
    sparse_pl = tf.placeholder(tf.float32, [1, 1], 'sparse_pl')
    p_act_enc = tf.reduce_mean(l_enc,0)
    reg_sparse = tf.reduce_sum(p_target * tf.log(p_target/p_act_enc) + (1-p_target)*tf.log((1-p_target)/(1-p_act_enc)))
    
    loss = loss_pure + sparse_pl * beta * reg_sparse
    
    
    
    # define our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=tau)
    
    # Manually define gradients
    gradients = tf.gradients(loss,params)
    gradients[0] -= alpha * relu(-params[0])
    gradients[2] -= alpha * relu(-params[2])
    train_op = optimizer.apply_gradients(zip(gradients,params))
    
    if not load_model:
        saver = tf.train.Saver() # we use this later to save the model


#%% test the forward pass
    _x_test = np.zeros(shape=(32, num_features))
    # initialize the Session
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
            
    feed_dict = {x_pl: _x_test}
    res_forward_pass = sess.run(fetches=[l_out], feed_dict=feed_dict)
    print("l_out", res_forward_pass[0].shape)


#%% Train
    full_weight_burn = False
    epoch_weight_burn = 0
    
    num_samples_train = x_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    #updates = []
    
    train_loss = []
    train_loss_pure = []
    train_sparse = []
    train_reg = []
    valid_loss = []
    sparse_weight = [[0]]
    sparse_counter = 0
    
    cur_loss = 0
    
    if do_plot:
        plt.figure(figsize=(10, 6))

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
                fetches_train = [train_op, loss, l_out,loss_pure,reg_sparse,reg_term]
                feed_dict_train = {x_pl: x_batch, sparse_pl: sparse_weight}
                # do the complete backprob pass
                res_train = sess.run(fetches_train, feed_dict_train)
                _, batch_loss, train_out,batch_loss_pure,batch_sparse,batch_reg = tuple(res_train)
                cur_loss += [batch_loss]
                cur_loss_pure += [batch_loss_pure]
                cur_sparse += [beta*batch_sparse]
                cur_reg += [alpha/2*batch_reg]
            
            
            train_sparse += [np.mean(cur_sparse)]
            train_reg += [np.mean(cur_reg)]
            train_loss += [np.mean(cur_loss)]
            train_loss[-1] += train_reg[-1]
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
#                sparse_weight[0][0] = np.logspace(log_start,log_end,n_weight_burn)[sparse_counter]
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
            
            # Plot progression while training
            # To be implemented later
            if do_plot and (epoch % 10 == 0 or epoch == 1):
                updates = [i*batch_size*num_batches_train for i in range(1,np.shape(train_loss)[0]+1)]
#                # Plotting
                plt.subplot(1,2,1)
                plt.title('Error')
                plt.xlabel('Updates'), plt.ylabel('Error')
                plt.semilogy(updates, train_loss, color="black")
                plt.semilogy(updates, train_loss_pure, color="red")
                plt.semilogy(updates, valid_loss, color="grey")
                plt.legend(['Train Error','Pure error', 'Valid Error'])
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.grid('on')
                
                plt.subplot(1,2,2)
                plt.title('Regularization')
                plt.xlabel('Updates'), plt.ylabel('Regularization')
                plt.semilogy(updates,train_sparse,color = "black")
                plt.semilogy(updates,train_reg,color = "gray")
                plt.legend(['Sparseness','Weight decay'])
                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
                plt.grid('on')

#                plt.subplot(num_classes+1,2,2)
#                plt.title('Regularization')
#                plt.xlabel('Updates'), plt.ylabel('Regularization')
#                plt.semilogy(updates,train_sparse,color = "black")
#                plt.semilogy(updates,train_reg,color = "gray")
#                plt.legend(['Sparseness','Weight decay'])
#                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#                plt.grid('on')
#                
#                c=0
#                for k in range(3, 3 + num_classes*2, 2):
#                    plt.subplot(num_classes+1,2,k)
#                    plt.cla()
#                    plt.title('Inputs for %i' % included_classes[c])
#                    plt.axis('off')
#                    idx = 0
#                    canvas = np.zeros((28*10, 10*28))
#                    for i in range(10):
#                        for j in range(10):
#                            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_valid[targets_valid==included_classes[c]][idx].reshape((28, 28))
#                            idx += 1
#                    plt.imshow(canvas, cmap='gray')
#                    
#                    plt.subplot(num_classes+1,2,k+1)
#                    plt.cla()
#                    plt.title('Reconstructions for %i' % included_classes[c])
#                    plt.axis('off')
#                    idx = 0
#                    canvas = np.zeros((28*10, 10*28))
#                    for i in range(10):
#                        for j in range(10):
#                            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = eval_out[targets_valid==included_classes[c]][idx].reshape((28, 28))
#                            idx += 1
#                    plt.imshow(canvas, cmap='gray')
#                    c+=1
#                plt.savefig("out51.png")
#                display(Image(filename="out51.png"))
#                clear_output(wait=True)
            
    except KeyboardInterrupt:
        pass

    if do_plot:
        plt.savefig('plots/' + plot_filename)
    return(sess,train_loss,train_loss_pure,valid_loss)
    
    
#%% Softmax layer
def Non_neg_softmax(x_train, x_valid,y_train,y_valid, **kwargs):          


#%% Unload check
    modelpath = kwargs.get('modelpath',None)
    alpha = kwargs.get('alpha',0.003)
    tau = kwargs.get('tau',0.001)
    num_epochs = kwargs.get('num_epochs',1000)
    batch_size = kwargs.get('batch_size',1000)
    eps = kwargs.get('eps',10**(-10))
    
    if not modelpath:
        load_model=False
    else:
        if isinstance(modelpath,str):
            load_model=True



#%% Define network
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
#%% Define loss function    
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


#%% test the forward pass
    _x_test = np.zeros(shape=(32, num_features))
    # initialize the Session
    sess = tf.Session()
    
    sess.run(tf.global_variables_initializer())
            
    feed_dict = {x_pl: _x_test}
    res_forward_pass = sess.run(fetches=[l_softmax], feed_dict=feed_dict)
    print("l_softmax", res_forward_pass[0].shape)


#%% Train 
    num_samples_train = x_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    #updates = []
    
    train_loss = []
    train_reg = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    
#    cur_loss = 0
    
    #if do_plot:
    #    plt.figure(figsize=(12, 24))
    
    try:
        for epoch in range(num_epochs):
            #Forward->Backprob->Update params
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
            
            
#            train_sparse += [np.mean(cur_sparse)]
            train_reg += [np.mean(cur_reg)]
            train_loss += [np.mean(cur_loss)]
            train_loss[-1] += train_reg[-1]
            train_acc += [np.mean(cur_acc)]
            
            # evaluate
            fetches_eval = [loss, reg_term, accuracy]
            feed_dict_eval = {x_pl: x_valid, y_pl: y_valid}                                
            res_valid = sess.run(fetches_eval, feed_dict_eval)
            eval_loss, eval_reg, eval_acc = tuple(res_valid)
#            print(eval_loss)
            valid_loss += [eval_loss]
            valid_acc += [eval_acc]
            

            print('Epoch %d. Train: %.5f. Train Acc: %.5f. Val: %.5f. Val Acc: %.5f. Weight: %.5f'
                  %(epoch+1,train_loss[-1],train_acc[-1],valid_loss[-1],np.mean(valid_acc[-1]),np.mean(cur_reg)))
            
            if epoch == 0:
                continue
            
            # Plot progression while training
            # To be implemented later
#            if do_plot and ((epoch+1) % 100 is 0 or (epoch+1) is num_epochs):
#                updates = [i*batch_size*num_batches_train for i in range(1,np.shape(train_loss)[0]+1)]
#                # Plotting
#                plt.subplot(num_classes+1,2,1)
#                plt.title('Error')
#                plt.xlabel('Updates'), plt.ylabel('Error')
#                plt.semilogy(updates, train_loss, color="black")
#                plt.semilogy(updates, train_loss_pure, color="red")
#                plt.semilogy(updates, valid_loss, color="grey")
#                plt.legend(['Train Error','Pure error', 'Valid Error'])
#                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#                plt.grid('on')
#                
#                plt.subplot(num_classes+1,2,2)
#                plt.title('Regularization')
#                plt.xlabel('Updates'), plt.ylabel('Regularization')
#                plt.semilogy(updates,train_sparse,color = "black")
#                plt.semilogy(updates,train_reg,color = "gray")
#                plt.legend(['Sparseness','Weight decay'])
#                plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
#                plt.grid('on')
#                
#                c=0
#                for k in range(3, 3 + num_classes*2, 2):
#                    plt.subplot(num_classes+1,2,k)
#                    plt.cla()
#                    plt.title('Inputs for %i' % included_classes[c])
#                    plt.axis('off')
#                    idx = 0
#                    canvas = np.zeros((28*10, 10*28))
#                    for i in range(10):
#                        for j in range(10):
#                            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = x_valid[targets_valid==included_classes[c]][idx].reshape((28, 28))
#                            idx += 1
#                    plt.imshow(canvas, cmap='gray')
#                    
#                    plt.subplot(num_classes+1,2,k+1)
#                    plt.cla()
#                    plt.title('Reconstructions for %i' % included_classes[c])
#                    plt.axis('off')
#                    idx = 0
#                    canvas = np.zeros((28*10, 10*28))
#                    for i in range(10):
#                        for j in range(10):
#                            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = eval_out[targets_valid==included_classes[c]][idx].reshape((28, 28))
#                            idx += 1
#                    plt.imshow(canvas, cmap='gray')
#                    c+=1
#                plt.savefig("out51.png")
#                display(Image(filename="out51.png"))
#                clear_output(wait=True)
            
    except KeyboardInterrupt:
        pass
    
    return(sess,train_loss,train_acc,valid_loss,valid_acc)
