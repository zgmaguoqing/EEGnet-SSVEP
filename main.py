import numpy as np

# mne imports
import mne
from mne import io
from mne.datasets import sample
from scipy.io import loadmat

# EEGNet-specific imports
from EEGModels import EEGNet, EEGNet_SSVEP
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# # PyRiemann imports
# from pyriemann.estimation import XdawnCovariances
# from pyriemann.tangentspace import TangentSpace
# from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

##################### Process, filter and epoch the data ######################
EEG_path = "/nfs/diskstation/DataStation/CheLiu/data/SSVEP/S1.mat"

# ‘Electrode index’, ‘Time points’, ‘Target index’, and ‘Block index’
EEG_data = loadmat(EEG_path)['data']  # [64, 1500, 40, 6]
EEG_data = EEG_data.transpose(3, 2, 0, 1)   # [6, 40, 64, 1500]

num_blocks, num_trials, chans, samples = EEG_data.shape
kernels = 1
# # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
# X = EEG_data * 1000  # format is in (trials, channels, samples)
# y = EEG_data
X = EEG_data  # [6, 40, 64, 1500]
y = np.array(list(range(num_trials)) * num_blocks).reshape(num_blocks, -1)   # [6, 40]

# train/validate/test
trainX = []
testX = []
trainY = []
trainY = []
acc_list = []
def cross_validation(X,y):
  KF = KFold(n_splits=6, shuffle=True, random=100)
  for index, (train_index, test_index) in enumerate(KF.split(X,y)):
        X_train, Y_train = X[train_index,...], y[train_index,...]
        X_test, Y_test = X[test_index,...], y[train_index,...]
        trainX.append(X_train)
        trainY.append(Y_train)
        testX.append(X_test)
        testY.append(Y_test)
  return np.array(trainX), np.array(trainY), np.array(testX), np.array(testY)

 for i in index:{
   
            print("fold-" + str(i + 1) + "mean acc: {}".format(np.mean(acc)))
          
        }

acc_list.append(np.mean(acc))
print("mean acc: {}".format(np.mean(acc_list)))
  
X_train = X[:5, ...].reshape(-1, chans, samples)
Y_train = y[:5, ...].reshape(-1)
# X_validate = X[144:216, ]
# Y_validate = y[144:216]
X_test = X[5, ...]
Y_test = y[5, :]

############################# EEGNet portion ##################################

# convert labels to one-hot encodings.
Y_train = np_utils.to_categorical(Y_train)
# Y_validate = np_utils.to_categorical(Y_validate)
Y_test = np_utils.to_categorical(Y_test)

# convert data to NHWC (trials, channels, samples, kernels) format. Data
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
# X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other
# model configurations may do better, but this is a good starting point)
# model = EEGNet(nb_classes=4, Chans=chans, Samples=samples,
#                dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
#                dropoutType='Dropout')
model = EEGNet_SSVEP(nb_classes=num_trials, Chans=chans, Samples=samples,
                     dropoutRate=0.5, kernLength=256, F1=96,
                     D=1, F2=96, dropoutType='Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

# count number of parameters in the model
numParams = model.count_params()

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='./tmp/checkpoint.h5', verbose=1,
                               save_best_only=False)

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during
# optimization to balance it out. This data is approximately balanced so we
# don't need to do this, but is shown here for illustration/completeness.
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0: 1, 1: 1, 2: 1, 3: 1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN +
# Riemannian geometry classification (below)
################################################################################
# fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=300,
#                         verbose=2, validation_data=(X_validate, Y_validate),
#                         callbacks=[checkpointer], class_weight=class_weights)
fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=100,
                        verbose=2, callbacks=[checkpointer])

# load optimal weights
model.load_weights('./tmp/checkpoint.h5')

###############################################################################
# can alternatively used the weights provided in the repo. If so it should get
# you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
# system.
###############################################################################

# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

'''
############################# PyRiemann Portion ##############################

# code is taken from PyRiemann's ERP sample script, which is decoding in
# the tangent space with a logistic regression

n_components = 2  # pick some components

# set up sklearn pipeline
clf = make_pipeline(XdawnCovariances(n_components),
                    TangentSpace(metric='riemann'),
                    LogisticRegression())

preds_rg = np.zeros(len(Y_test))

# reshape back to (trials, channels, samples)
X_train = X_train.reshape(X_train.shape[0], chans, samples)
X_test = X_test.reshape(X_test.shape[0], chans, samples)

# train a classifier with xDAWN spatial filtering + Riemannian Geometry (RG)
# labels need to be back in single-column format
clf.fit(X_train, Y_train.argmax(axis=-1))
preds_rg = clf.predict(X_test)

# Printing the results
acc2 = np.mean(preds_rg == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc2))

# plot the confusion matrices for both classifiers
names = ['audio left', 'audio right', 'vis left', 'vis right']
plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis=-1), names, title='EEGNet-8,2')

plt.figure(1)
plot_confusion_matrix(preds_rg, Y_test.argmax(axis=-1), names, title='xDAWN + RG')
'''
