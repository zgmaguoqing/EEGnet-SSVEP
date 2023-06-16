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
from Data_loader import Data_loader

# # PyRiemann imports
# from pyriemann.estimation import XdawnCovariances
# from pyriemann.tangentspace import TangentSpace
# from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

total_info = []
for sub in range(1, 6):
    EEG_path = "/nfs/diskstation/DataStation/CheLiu/data/SSVEP/S%d.mat" % sub
    data_loader=Data_loader(windows_width=300,windows_step=50,num_trials=10, EEG_path=EEG_path) # 使用windows_width=1500-1,windows_step=1时，相当于没有滑动窗口处理。num_trials表示取40类中的几个类别。

    # num_blocks, num_trials, chans, samples = data_loader.EEG_data.shape
    acc_list = []
    for cv, (X_train, Y_train, X_test, Y_test) in enumerate(data_loader.get_data()):
        print('------------------------------------')
        print('CV %d begins:' % (cv + 1))
        model = EEGNet_SSVEP(nb_classes=data_loader.num_trials, Chans=data_loader.chans, Samples=data_loader.windows_width,
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
        result_list = []
        for i in range(100): # train 10 epochs
            ################################################################################
            # fit the model. Due to very small sample sizes this can get
            # pretty noisy run-to-run, but most runs should be comparable to xDAWN +
            # Riemannian geometry classification (below)
            ################################################################################
            # fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=300,
            #                         verbose=2, validation_data=(X_validate, Y_validate),
            #                         callbacks=[checkpointer], class_weight=class_weights)
            fittedModel = model.fit(X_train, Y_train, batch_size=16, epochs=1,
                                    verbose=2, callbacks=[checkpointer])

            # load optimal weights
            #model.load_weights('./tmp/checkpoint.h5')

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

            probs = model(X_test).numpy()
            preds = probs.argmax(axis=-1)
            print('pred: ', preds)
            acc = np.mean(preds == Y_test.argmax(axis=-1))
            result_list.append(acc)
            print("Epoch %d: Classification accuracy: %f " % (i+1, acc))
        print('\n----------------')
        best_result = np.max(result_list)
        acc_list.append(best_result)
        print('Best result: %f' % best_result)
        print('------------------------------------\n\n\n')

    print('\n------------------- s%d -------------------------------------------------' % sub)
    acc_min = np.min(acc_list)
    acc_max = np.max(acc_list)
    acc_mean = np.mean(acc_list)
    acc_list.append(acc_min)
    acc_list.append(acc_max)
    acc_list.append(acc_mean)
    row_str = 'S%d & ' % sub + ' & '.join([str(np.round(accuracy * 100, 2)) for accuracy in acc_list])
    total_info.append(row_str)
    print(row_str)
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------')
    print('------------------------------------------------------------------------\n')

print('\n\n\n\n\n-----------------------------TOTAL--------------------------------------')
print('------------------------------------------------------------------------')
print('\n'.join(total_info))

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
