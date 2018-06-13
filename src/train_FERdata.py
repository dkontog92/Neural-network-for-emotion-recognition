import numpy as np
from matplotlib.pyplot import imread
import pandas as pd
from test_metrics import *
from fcnet import FullyConnectedNet
from utils.solver import Solver
import pickle
import os

def get_FER2013_data(datadir, labels, validation_ratio=None, mirror=True):
    train_id = np.array([True if file[:5] == 'Train' else False for file in labels['img']])
    
    print("Loading data...")

    xtrain = np.array([imread(datadir + '/' + name)[:, :, 0] for name in labels['img'][train_id]])
    mean = xtrain.mean()
   
    xtrain = xtrain - mean
    ytrain = labels['emotion'][train_id].values

    print("Train data are of size %s" % str(xtrain[0].shape))

    xtest = np.array([imread(datadir + '/' + name)[:, :, 0] for name in labels['img'][~train_id]])
    # xtest = (xtest - mean) / np.sqrt(var)
    xtest = xtest - mean
    ytest = labels['emotion'][~train_id].values

    if validation_ratio:
        num_data = xtrain.shape[0]

        num_data_val = np.floor(validation_ratio * num_data).astype(int)
        val_data_indices = np.array([True]*num_data_val + [False]*(num_data - num_data_val))
        # np.random.shuffle(val_data_indices)

        xval = xtrain[val_data_indices]
        yval = ytrain[val_data_indices]

        xtrain = xtrain[~val_data_indices]
        ytrain = ytrain[~val_data_indices]
    else:
        xval = None
        yval = None
    
    if mirror:
        xtrain_mirror = np.fliplr(xtrain)
        xtrain = np.concatenate((xtrain, xtrain_mirror), axis=0)
        ytrain = np.concatenate((ytrain, ytrain))
    
    data = {'X_train': xtrain, 'y_train': ytrain,
            'X_val': xval, 'y_val': yval,
            'X_test': xtest, 'y_test': ytest}
    
    return data, mean

os.chdir("..\\datasets\\public")
datadir = os.getcwd()
labels = pd.read_csv(datadir + '\\labels_public.txt')


data, mean = get_FER2013_data(datadir, labels, validation_ratio=0.10,mirror = False)

input_dim = 2304
out_dim = 7

#PARAMETERS 
#These parameters were changed to optimize the performance of the network. For each parameter, n different values were given in a list, and the neural network was trained n times within a loop in order to observe the parameter value that makes th network perform best in the validation set 
hidden_dims = [512, 256]
learning_rate = 1e-2
p_of_drop = 0.5
regul = 0.001
momentum = 0.9
batch = 150
epochs = 50
lr_decay = 0.95


net = FullyConnectedNet(hidden_dims=hidden_dims, input_dim=input_dim,
                        num_classes=out_dim, dropout=p_of_drop, reg=regul, seed=0)
            
solver = Solver(net,
                data,
                update_rule = 'sgd_momentum',
                optim_config={'learning_rate': learning_rate,
                              'momentum': momentum},
                lr_decay=lr_decay,
                num_epochs=epochs,
                batch_size=batch,
                print_every=1000)
            
solver.train()

make_plots = True
if make_plots:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('Iteration')
    plt.subplot(2, 1, 2)
    plt.title('Accuracy')
    plt.plot(solver.train_acc_history, '-o', label='train')
    plt.plot(solver.val_acc_history, '-o', label='val')
    plt.plot([0.5] * len(solver.val_acc_history), 'k--')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.gcf().set_size_inches(15, 12)
    plt.show()

