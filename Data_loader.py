import torch
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import KFold
from tensorflow.keras import utils as np_utils


class BaseData_loader:
    def __init__(self, n_input, n_data):
        pass
        self.train_data = self.generate_train_data(n_input, n_data)
        self.test_data = self.generate_test_data(n_input, n_data)

    def generate_train_data(self, num_input, n_data, train_index=None):
        train_data = torch.rand(n_data, num_input)
        return train_data

    def generate_test_data(self, num_input, n_data, test_index=None):
        test_data = torch.rand(n_data, num_input)
        return test_data


class Data_loader(BaseData_loader):
    def __init__(self, EEG_path, n_input=0, n_data=0, windows_width=500, windows_step=20, num_trials=10):
        # super().__init__(n_input=n_input,n_data=n_data)
        self.windows_width = windows_width
        self.windows_step = windows_step
        self.num_trials = num_trials
        ##################### Process, filter and epoch the data ######################
        # ‘Electrode index’, ‘Time points’, ‘Target index’, and ‘Block index’
        EEG_data = loadmat(EEG_path)['data']  # [64, 1500, 40, 6]
        EEG_data = EEG_data.transpose(3, 2, 0, 1)  # [6, 40, 64, 1500]

        num_blocks, num_trials, self.chans, self.samples = EEG_data.shape
        num_trials = self.num_trials
        self.kernels = 1
        # # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
        # X = EEG_data * 1000  # format is in (trials, channels, samples)
        # y = EEG_data
        channel_list = [60, 61, 62, 47, 53, 54, 55, 56, 57]
        self.X = EEG_data[:, :num_trials, channel_list, :]  # [6, 10, 9, 1500]
        self.chans = 9
        self.y = np.array(list(range(num_trials)) * num_blocks).reshape(num_blocks, -1)  # [6, 40]

    def get_data(self):
        kf = KFold(n_splits=6, shuffle=False)  # 初始化KFold
        data = []
        for train_index, test_index in kf.split(self.X):  # 调用split方法切分数据
            # print('train_index:%s , test_index: %s ' % (train_index, test_index))
            X_train, Y_train = self.generate_train_data(num_input=0, n_data=0, train_index=train_index)
            X_test, Y_test = self.generate_test_data(num_input=0, n_data=0, test_index=test_index)
            data.append((X_train, Y_train, X_test, Y_test))
        return data

    def generate_train_data(self, num_input, n_data, train_index=None):
        samples = self.samples

        # train/validate/test
        X_train_data = self.X[train_index].reshape(-1, self.chans, samples)  # [200, 64, 1500]
        Y_train_data = self.y[train_index].reshape(-1)  # [200]
        X_train = np.zeros(
            (X_train_data.shape[0], int((samples - self.windows_width) / self.windows_step), self.chans, self.windows_width))
        Y_train = np.zeros((Y_train_data.shape[0], int((samples - self.windows_width) / self.windows_step)))
        for i in range(X_train_data.shape[0]):
            for index, start_index in enumerate(
                    [x * self.windows_step for x in range(int((samples - self.windows_width) / self.windows_step))]):
                X_train[i, index, :, :] = X_train_data[i, :, start_index:start_index + self.windows_width]
                Y_train[i, index] = Y_train_data[i]
        X_train = X_train.reshape(-1, self.chans, self.windows_width)  # [200*100, 64, 500]
        noise = np.random.normal(0, 1.0, size=X_train.shape).astype(np.float32)
        percentage = 0.0
        mask = np.random.choice([0, 1], size=X_train.shape, p=[1 - percentage, percentage])
        X_train += noise * mask
        Y_train = Y_train.reshape(-1)  # [200*100]
        ############################# EEGNet portion ##################################

        # convert labels to one-hot encodings.
        Y_train = np_utils.to_categorical(Y_train)  # [200*100,40]

        # convert data to NHWC (trials, channels, samples, kernels) format. Data
        # contains 60 channels and 151 time-points. Set the number of kernels to 1.
        X_train = X_train.reshape(X_train.shape[0], self.chans, self.windows_width, self.kernels)  # [200*100, 64, 500,1]

        # print('X_train shape:', X_train.shape)  # [200*100, 64, 500,1] 200表示5个block乘40个类别
        # print(X_train.shape[0], 'train samples')

        return X_train, Y_train

    def generate_test_data(self, num_input, n_data, test_index=None):
        X_test_data = self.X[test_index].reshape(-1, self.chans, self.samples)  # [40,64,500]
        Y_test_data = self.y[test_index].reshape(-1)  # [40]
        # X_test = np.zeros((X_test_data.shape[0], int((samples - self.windows_width) / self.windows_step), chans, self.windows_width))
        # Y_test = np.zeros((Y_test_data.shape[0], int((samples - self.windows_width) / self.windows_step)))
        # for i in range(X_test_data.shape[0]):
        #     for index, start_index in enumerate(
        #             [x * self.windows_step for x in range(int((samples - self.windows_width) / self.windows_step))]):
        #         X_test[i, index, :, :] = X_test_data[i, :, start_index:start_index + self.windows_width]
        #         Y_test[i, index] = Y_test_data[i]
        # X_test = X_test.reshape(-1, chans, self.windows_width)  # [40*100, 64, 500]
        # Y_test = Y_test.reshape(-1)  # [40*100]
        X_test = X_test_data[:, :, self.samples - self.windows_width:self.samples]
        Y_test = Y_test_data
        ############################# EEGNet portion ##################################

        # Y_validate = np_utils.to_categorical(Y_validate)
        Y_test = np_utils.to_categorical(Y_test)

        # X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
        X_test = X_test.reshape(X_test.shape[0], self.chans, self.windows_width, self.kernels)

        # print(X_test.shape[0], 'test samples')

        return X_test, Y_test

    def data_generator(self, n_data):  # z是自变量 ，y是因变量。y: dim*samples
        ##################### Process, filter and epoch the data ######################
        EEG_path = "./data/SSVEP/S1.mat"  # "/nfs/diskstation/DataStation/CheLiu/data/SSVEP/S1.mat"

        # ‘Electrode index’, ‘Time points’, ‘Target index’, and ‘Block index’
        EEG_data = loadmat(EEG_path)['data']  # [64, 1500, 40, 6]
        EEG_data = EEG_data.transpose(3, 2, 0, 1)  # [6, 40, 64, 1500]

        num_blocks, num_trials, chans, samples = EEG_data.shape
        kernels = 1
        # # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
        # X = EEG_data * 1000  # format is in (trials, channels, samples)
        # y = EEG_data
        X = EEG_data  # [6, 40, 64, 1500]
        y = np.array(list(range(num_trials)) * num_blocks).reshape(num_blocks, -1)  # [6, 40]

        # train/validate/test
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
        return (z, y)


if __name__ == '__main__':
    data_loader = Data_loader(num_input=1,
                              n_data=50)
    data_loader.generate_train_data
