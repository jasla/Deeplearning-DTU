

import matplotlib.pyplot as plt
import numpy as np
import itertools
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