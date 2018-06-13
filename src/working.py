import numpy as np
from src.utils.data_utils import *
from src.utils.solver import *
from src.fcnet import *

datadir = ('/home/mat10/Documents/MSc Machine Learning/395-Machine Learning/'
           'CW2/assignment2_advanced/datasets/cifar-10-batches-py/')

traindata = load_CIFAR_batch(datadir + 'data_batch_1')
testdata = load_CIFAR_batch(datadir + 'test_batch')

# X = data[0][0:10]
# X.shape
#
# N = 10
# M = 100
# D = np.prod(X[0].shape)
#
# W = np.random.randn(D, M)
# b = np.random.randn(M)
#
# out = linear_forward(X, W, b)
# out.shape
#
# #X = out.copy()
# dout = np.random.randn(N, M)
# dout.shape
#
# dX, dW, db = linear_backward(dout, X, W, b)

hidden_dims = [1024, 512]

N = 50

net = FullyConnectedNet(hidden_dims, num_classes=10,
                 dropout=0., reg=0.0)
#
# X = data[0][0:N]
# scores = net.loss(X)
#
# y = np.random.choice(10, N)
#
# loss, grads = net.loss(X, y)

data = {
      'X_train': traindata[0][:N],
      'y_train': traindata[1][:N],
      'X_val': testdata[0],
      'y_val': testdata[1]
       }

solver = Solver(net,
                data,
                update_rule='sgd',
                optim_config={'learning_rate': 1e-3},
                lr_decay=0.95,
                num_epochs=20,
                batch_size=10,
                print_every=100)
solver.train()




import numpy as np
from scipy.ndimage import imread
import os
import pandas as pd

def get_jpg_only(datapath):
    """
    imports all .jpg files from a directory and returns them along with file names
    """
    names = [file for file in os.listdir(datapath) if file[-3:] == 'jpg' ]
    data = np.array([imread(datapath + '/' + name)[:, :, 0].ravel() for name in names])
    return data, names


def get_FER2013_data(datadir, labels, validation_ratio=0.2):
    train_id = np.array([True if file[:5] == 'Train' else False for file in labels['img']])
    xtrain = np.array([imread(datadir + '/' + name)[:, :, 0].ravel() for name in labels['img'][train_id]])
    ytrain = labels['emotion'][train_id]
    xtest = np.array([imread(datadir + '/' + name)[:, :, 0].ravel() for name in labels['img'][~train_id]])
    ytest = labels['emotion'][~train_id]
    if validation_ratio:
        num_data = xtrain.shape[0]

        num_data_val = np.floor(validation_ratio * num_data).astype(int)
        val_data_indices = np.array([True]*num_data_val + [False]*(num_data - num_data_val))
        np.random.shuffle(val_data_indices)

        xval = xtrain[val_data_indices]
        yval = ytrain[val_data_indices]

        xtrain = xtrain[~val_data_indices]
        ytrain = ytrain[~val_data_indices]
    else:
        xval = None
        yval = None
    return xtrain, ytrain, xval, yval, xtest, ytest


datadir = ("/home/mat10/Documents/MSc Machine Learning/"
           "395-Machine Learning/CW2/assignment2_advanced/"
           "datasets/FER2013")

labels = pd.read_csv(datadir + '/labels_public.txt')
labels.shape
labels.columns


validation_ratio = 0.2
xtrain, ytrain, xval, yval, xtest, ytest = get_FER2013_data(datadir, labels, validation_ratio)






datapath = datadir + '/Train'

data = imread(datapath + '/' + names[0])[:, :, 0]
data.shape




arr = np.array([[1,1,1], [2,2,2], [3,3,3]])
np.random.shuffle(np.arange(100))
arr

train_data_files = [file for file in os.listdir(datadir + "/Train") if file[-3:] == 'jpg' ]

sum([file[-3:] == 'jpg' for file in train_data_files])





a = np.array([[1,1,1,1], [2,2,2,2], [3,3,3,3]])

a/a.sum(axis=1)[:, None]


