import cupy as np
import numpy
import traceback
import pandas as pd
import pickle
import os
from tqdm import tqdm
import cv2
import matplotlib as plt
import matplotlib.patches as patches
import tensorflow as tf

try:
    print(f"CuPy đang chạy trên thiết bị: {np.cuda.Device(0).use()}")
    gpu_mode = True
except Exception as e:
    print("CẢNH BÁO: Không tìm thấy GPU hoặc chưa cài CuPy đúng cách. Đang chạy fallback...")
    gpu_mode = False

def PreProcess():
    path_imgs = r'C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\Project\DL\Convolutional network\Obj_Detection\Raccoon_detection\Racoon_Images\images'
    labels = pd.read_csv('train_labels_.csv')
    new_size = 256
    X_train = []
    X_valid = []
    X_test = []
    y_test_org = numpy.asarray(labels[['filename','width','height','xmin','ymin','xmax','ymax']].iloc[160:].values)
    y_valid_org = numpy.asarray(labels[['filename','width','height','xmin','ymin','xmax','ymax']].iloc[120:160].values)

    for path in tqdm(labels['filename'].values[:120]):
        org_img = cv2.imread(path_imgs + "/" + path)
        if org_img is not None:
            img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (new_size, new_size))
            X_train.append(img)

    for path in tqdm(labels['filename'].values[120:160]):
        org_img = cv2.imread(path_imgs + "/" + path)
        if org_img is not None:
            img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (new_size, new_size))
            X_valid.append(img)

    for path in tqdm(labels['filename'].values[160:]):
        org_img = cv2.imread(path_imgs + "/" + path)
        if org_img is not None:
            img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (new_size, new_size))
            X_test.append(img)

    labels['xmin'] = labels['xmin'] / labels['width'] * new_size
    labels['xmax'] = labels['xmax'] / labels['width'] * new_size
    labels['ymin'] = labels['ymin'] / labels['height'] * new_size
    labels['ymax'] = labels['ymax'] / labels['height'] * new_size
    cols =['xmin','ymin','xmax','ymax']

    X_train = numpy.array(X_train)
    X_train = X_train / 255
    X_valid = numpy.array(X_valid)
    X_valid = X_valid / 255
    X_test = numpy.array(X_test)
    X_test = X_test / 255
    y_train = numpy.asarray(labels[cols].iloc[:120].values).astype(numpy.float32)
    y_valid = numpy.asarray(labels[cols].iloc[120:160].values).astype(numpy.float32)
    y_test = numpy.asarray(labels[cols].iloc[160:].values).astype(numpy.float32)

    return X_train, X_valid, X_test, y_train, y_valid, y_test, y_test_org, y_valid_org


def Adam(W, B, dW, dB, V_dw, V_db, S_dw, S_db, Beta1, Beta2, num_of_mn_batch,lr):
    num_of_mn_batch += 1

    V_dw = Beta1 * V_dw + (1 - Beta1) * dW
    V_db = Beta1 * V_db + (1 - Beta1) * dB
    S_dw = Beta2 * S_dw + (1 - Beta2) * (dW ** 2)
    S_db = Beta2 * S_db + (1 - Beta2) * (dB ** 2)

    V_dw_corr = V_dw / (1 - Beta1 ** num_of_mn_batch + 1e-8)
    V_db_corr = V_db / (1 - Beta1 ** num_of_mn_batch + 1e-8)
    S_dw_corr = S_dw / (1 - Beta2 ** num_of_mn_batch + 1e-8)
    S_db_corr = S_db / (1 - Beta2 ** num_of_mn_batch + 1e-8)

    W = W - lr * V_dw_corr / (np.sqrt(S_dw_corr) + 1e-8)
    B = B - lr * V_db_corr / (np.sqrt(S_db_corr) + 1e-8)

    return W, B, V_dw, V_db, S_dw, S_db


class Conv_layer:
    # Initialize
    def __init__(self, filters=None, kernel_size=3, pad=0, strides=1, padding_mode=None, learning_rate=0.0003,
                 epsilon=10 ** -8, Beta1=0.9, Beta2=0.999, decay_rate=0.95):
        self.filters = filters
        self.kernel_size = kernel_size
        self.pad = pad
        self.strides = strides
        self.padding_mode = padding_mode
        self.lr = learning_rate
        self.epsilon = epsilon
        self.V_dw = 0
        self.V_db = 0
        self.S_dw = 0
        self.S_db = 0
        self.Beta1 = Beta1
        self.Beta2 = Beta2
        self.decay_rate = decay_rate
        self.W = None
        self.B = None
        self.output = None
        self.cache = None

    # Forward
    def zero_pad(self, X, pad):
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        return X_pad

    def conv_forward(self, A_prev, pad, strides):
        (m, n_H_prev, n_W_prev, channels) = A_prev.shape
        (f, f, channels, n_filters) = self.W.shape

        n_H = int((n_H_prev + 2 * pad - f) / strides) + 1
        n_W = int((n_W_prev + 2 * pad - f) / strides) + 1

        A_prev_pad = self.zero_pad(A_prev, pad)

        s0, s1, s2, s3 = A_prev_pad.strides
        new_shape = (m, n_H, n_W, f, f, channels)
        new_strides = (s0, s1 * strides, s2 * strides, s1, s2, s3)

        A_col_view = np.lib.stride_tricks.as_strided(A_prev_pad, shape=new_shape, strides=new_strides)
        A_col = A_col_view.reshape(m * n_H * n_W, -1)
        W_col = self.W.reshape(-1, n_filters)

        Z_col = np.dot(A_col, W_col) + self.B.reshape(1, n_filters)
        Z = Z_col.reshape(m, n_H, n_W, n_filters)

        cache = (A_prev, self.W, self.B, pad, strides)

        return Z, cache

    def pre_process(self, data):
        if self.W is None:
            fan_in = self.kernel_size * self.kernel_size * data.shape[3]
            scale = np.sqrt(2.0 / fan_in)
            self.W = np.random.randn(self.kernel_size, self.kernel_size, data.shape[3], self.filters).astype(np.float32) * scale
            self.B = np.zeros((1, 1, 1, self.filters), dtype = np.float32)

        if self.padding_mode == "valid":
            self.pad = 0
            self.strides = 1
        elif self.padding_mode == "same":
            self.pad = int((self.kernel_size - 1) / 2)
            self.strides = 1

        pad = self.pad
        strides = self.strides

        A_prev = data
        A, cache = self.conv_forward(A_prev, pad, strides)

        return A, cache

    def forward(self, input):
        self.output, self.cache = self.pre_process(input)
        return self.output

    # Backward
    def backward(self, dZ, num_of_mn_batch, epoch):
        (A_prev, W, B, pad, strides) = self.cache
        (m, n_H, n_W, n_C) = dZ.shape
        (f, f, pre_channels, n_filters) = W.shape

        dB = np.sum(dZ, axis=(0, 1, 2)).reshape(1, 1, 1, n_filters)

        A_prev_pad = self.zero_pad(A_prev, pad)
        s0, s1, s2, s3 = A_prev_pad.strides
        new_shape = (m, n_H, n_W, f, f, pre_channels)
        new_strides = (s0, s1 * strides, s2 * strides, s1, s2, s3)

        A_col_view = np.lib.stride_tricks.as_strided(A_prev_pad, shape=new_shape, strides=new_strides)
        A_col = A_col_view.reshape(m * n_H * n_W, -1)
        dZ_col = dZ.reshape(-1, n_filters)

        dW_col = np.dot(A_col.T, dZ_col)
        dW = dW_col.reshape(f, f, pre_channels, n_filters)

        W_reshape = W.reshape(-1, n_filters)
        dA_col = np.dot(dZ_col, W_reshape.T)
        dA_col_reshaped = dA_col.reshape(m, n_H, n_W, f, f, pre_channels)

        dA_prev_pad = np.zeros_like(A_prev_pad)

        for i in range(f):
            for j in range(f):
                dA_prev_pad[:, i:i + n_H * strides:strides, j:j + n_W * strides:strides, :] += dA_col_reshaped[
                    :, :, :, i, j, :]

        if pad != 0:
            dA_prev = dA_prev_pad[:, pad:-pad, pad:-pad, :]
        else:
            dA_prev = dA_prev_pad

        lr = self.lr * (self.decay_rate ** epoch)
        (self.W, self.B, self.V_dw, self.V_db, self.S_dw, self.S_db) = Adam(self.W, self.B,dW, dB, self.V_dw,self.V_db,
                                                                            self.S_dw,self.S_db,self.Beta1, self.Beta2,
                                                                            num_of_mn_batch,lr)
        return dA_prev


class Pooling_layer:
    # Initialize
    def __init__(self, kernel_size=None, strides=None, pooling_mode="max"):
        self.kernel_size = kernel_size
        self.strides = strides
        self.pooling_mode = pooling_mode
        self.output = None
        self.cache = None
    # Forward
    def pooling_forward(self, A_prev, f, strides):
        (m, n_H_prev, n_W_prev, n_C) = A_prev.shape
        n_H = int((n_H_prev - f) / strides) + 1
        n_W = int((n_W_prev - f) / strides) + 1

        s0, s1, s2, s3 = A_prev.strides
        new_shape = (m, n_H, n_W, f, f, n_C)
        new_strides = (s0, s1 * strides, s2 * strides, s1, s2, s3)

        A_prev_view = np.lib.stride_tricks.as_strided(
            A_prev, shape=new_shape, strides=new_strides
        )

        if self.pooling_mode == "max":
            A = np.max(A_prev_view, axis=(3, 4))
        elif self.pooling_mode == "average":
            A = np.mean(A_prev_view, axis=(3, 4))
        else:
            return None, None

        cache = (A_prev, f, strides)
        return A, cache

    def forward(self, A_prev):
        self.output, self.cache = self.pooling_forward(A_prev, self.kernel_size, self.strides)
        return self.output

    # Backward
    def backward(self, dA):
        (A_prev, f, strides) = self.cache
        (m, n_H, n_W, n_C) = dA.shape
        dA_prev = np.zeros(A_prev.shape)

        if self.pooling_mode == "max":
            s0, s1, s2, s3 = A_prev.strides
            new_shape = (m, n_H, n_W, f, f, n_C)
            new_strides = (s0, s1 * strides, s2 * strides, s1, s2, s3)
            A_prev_windows = np.lib.stride_tricks.as_strided(A_prev, shape=new_shape, strides=new_strides)

            max_val = np.max(A_prev_windows, axis=(3, 4), keepdims=True)
            mask = (A_prev_windows == max_val)

            dA_expanded = dA.reshape(m, n_H, n_W, 1, 1, n_C)
            d_window = mask * dA_expanded

            for i in range(f):
                for j in range(f):
                    dA_prev[:, i:i + n_H * strides:strides, j:j + n_W * strides:strides, :] += d_window[
                        :, :, :, i, j, :]

        elif self.pooling_mode == "average":
            da = dA / (f * f)
            for i in range(f):
                for j in range(f):
                    dA_prev[:, i:i + n_H * strides:strides, j:j + n_W * strides:strides, :] += da

        return dA_prev


class BatchNorm:
    # Initialize
    def __init__(self, eps=1e-05, gamma=1, beta=0, mean=None, var=None, learning_rate=0.0003,
                 epsilon=10 ** -8, Beta1=0.9, Beta2=0.999, decay_rate=0.95):
        self.eps = eps
        self.gamma = gamma
        self.beta = beta
        self.mean = mean
        self.var = var
        self.lr = learning_rate
        self.epsilon = epsilon
        self.V_dGamma = 0
        self.V_dBeta = 0
        self.S_dGamma = 0
        self.S_dBeta = 0
        self.Beta1 = Beta1
        self.Beta2 = Beta2
        self.decay_rate = decay_rate
        self.output = None
        self.cache = None

    # Forward
    def forward(self, data, training = False):
        if training == True:
            self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True)
            self.var = np.var(data, axis=(0, 1, 2), keepdims=True)
        x_hat = (data - self.mean) / (np.sqrt(self.var + self.eps))
        y = self.gamma * x_hat + self.beta
        self.cache = (x_hat, data)

        return y

    # Backward
    def backward(self, dZ, num_of_mn_batch, epoch):
        X_hat, X = self.cache
        bias = X - self.mean
        temp = self.var + self.eps
        m = X.shape[0] * X.shape[1] * X.shape[2]

        dX_hat = dZ * self.gamma
        dBeta = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
        dGamma = np.sum(dZ * X_hat, axis=(0, 1, 2), keepdims=True)
        dVar = np.sum(dX_hat * bias * -1 / 2 * np.power(temp, -3 / 2), axis=(0, 1, 2), keepdims=True)
        dMean = np.sum(dX_hat * -1 / np.sqrt(temp) + dVar * -2 * bias / m, axis=(0, 1, 2), keepdims=True)
        dX = dVar * 2 * bias / m + dMean / m + dX_hat / np.sqrt(temp)

        lr = self.lr * (self.decay_rate ** epoch)
        (self.gamma, self.beta, self.V_dGamma, self.V_dBeta, self.S_dGamma, self.S_dBeta) = Adam(self.gamma,self.beta,
                                                                                                 dGamma, dBeta,self.V_dGamma,
                                                                                                 self.V_dBeta,self.S_dGamma,
                                                                                                 self.S_dBeta, self.Beta1,
                                                                                                 self.Beta2,num_of_mn_batch,lr)
        return dX


class ReLU():
    def __init__(self):
        self.threshold = 0.0
        self.cache = None

    def ReLU(self, data):
        self.cache = np.copy(data)
        data[data < self.threshold] = data[data < self.threshold] * 0.01
        return data

    def forward(self, data):
        self.output = self.ReLU(data)
        return self.output

    def backward(self, dZ):
        X = self.cache
        dA = np.copy(dZ)
        dA[X <= self.threshold] = dA[X <= self.threshold] * 0.01
        return dA


class Identity_block:
    def __init__(self, kernel_size=3, filters=None):
        self.kernel_size = kernel_size
        self.filters = filters
        self.conv_layer1 = Conv_layer(self.filters[0], 1, padding_mode="same")
        self.BN_layer1 = BatchNorm()
        self.ReLU_layer1 = ReLU()
        self.conv_layer2 = Conv_layer(self.filters[1], self.kernel_size, padding_mode="same")
        self.BN_layer2 = BatchNorm()
        self.ReLU_layer2 = ReLU()
        self.conv_layer3 = Conv_layer(self.filters[2], 1, padding_mode="same")
        self.BN_layer3 = BatchNorm()
        self.ReLU_layer3 = ReLU()

    def forward(self, data, training = False):
        X = data
        X_shortcut = X

        X = self.conv_layer1.forward(X)
        X = self.BN_layer1.forward(X, training)
        X = self.ReLU_layer1.forward(X)

        X = self.conv_layer2.forward(X)
        X = self.BN_layer2.forward(X, training)
        X = self.ReLU_layer2.forward(X)

        X = self.conv_layer3.forward(X)
        X = self.BN_layer3.forward(X, training)

        X = X + X_shortcut
        X = self.ReLU_layer3.forward(X)

        return X

    def backward(self, dZ, num_of_mn_batch, epoch):
        dX = self.ReLU_layer3.backward(dZ)
        dX = self.BN_layer3.backward(dX,num_of_mn_batch, epoch)
        dX = self.conv_layer3.backward(dX, num_of_mn_batch, epoch)

        dX = self.ReLU_layer2.backward(dX)
        dX = self.BN_layer2.backward(dX, num_of_mn_batch, epoch)
        dX = self.conv_layer2.backward(dX, num_of_mn_batch, epoch)

        dX = self.ReLU_layer1.backward(dX)
        dX = self.BN_layer1.backward(dX, num_of_mn_batch, epoch)
        dX = self.conv_layer1.backward(dX, num_of_mn_batch, epoch)

        dX_shortcut = self.ReLU_layer3.backward(dZ)
        dX += dX_shortcut

        return dX


class Conv_block:
    def __init__(self, kernel_size=3, filters=None, strides=2):
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.conv_layer1 = Conv_layer(self.filters[0], 1, strides=self.strides)
        self.BN_layer1 = BatchNorm()
        self.ReLU_layer1 = ReLU()
        self.conv_layer2 = Conv_layer(self.filters[1], self.kernel_size, padding_mode="same")
        self.BN_layer2 = BatchNorm()
        self.ReLU_layer2 = ReLU()
        self.conv_layer3 = Conv_layer(self.filters[2], 1, padding_mode="valid")
        self.BN_layer3 = BatchNorm()
        self.ReLU_layer3 = ReLU()
        self.conv_shortcut = Conv_layer(self.filters[2], 1, strides=self.strides)
        self.BN_shortcut = BatchNorm()

    def forward(self, data, training = False):
        X = data
        X_shortcut = X

        X = self.conv_layer1.forward(X)
        X = self.BN_layer1.forward(X, training)
        X = self.ReLU_layer1.forward(X)

        X = self.conv_layer2.forward(X)
        X = self.BN_layer2.forward(X, training)
        X = self.ReLU_layer2.forward(X)

        X = self.conv_layer3.forward(X)
        X = self.BN_layer3.forward(X, training)

        X_shortcut = self.conv_shortcut.forward(X_shortcut)
        X_shortcut = self.BN_shortcut.forward(X_shortcut, training)

        X = X + X_shortcut
        X = self.ReLU_layer3.forward(X)
        return X

    def backward(self, dZ, num_of_mn_batch, epoch):
        dX = self.ReLU_layer3.backward(dZ)
        dX = self.BN_layer3.backward(dX, num_of_mn_batch, epoch)
        dX = self.conv_layer3.backward(dX, num_of_mn_batch, epoch)

        dX = self.ReLU_layer2.backward(dX)
        dX = self.BN_layer2.backward(dX, num_of_mn_batch, epoch)
        dX = self.conv_layer2.backward(dX, num_of_mn_batch, epoch)

        dX = self.ReLU_layer1.backward(dX)
        dX = self.BN_layer1.backward(dX, num_of_mn_batch, epoch)
        dX = self.conv_layer1.backward(dX, num_of_mn_batch, epoch)

        dX_shortcut = self.ReLU_layer3.backward(dZ)
        dX_shortcut = self.BN_shortcut.backward(dX_shortcut, num_of_mn_batch, epoch)
        dX_shortcut = self.conv_shortcut.backward(dX_shortcut, num_of_mn_batch, epoch)

        dX += dX_shortcut
        return dX


class AVG_pooling_global:
    def __init__(self):
        self.output = None
        self.cache = None

    def forward(self, data):
        self.output = np.mean(data, axis=(1, 2), keepdims=True)
        self.cache = data.shape
        return self.output

    def backward(self, dZ):
        (m, n_H, n_W, n_C) = self.cache
        dA_filled = dZ / (n_H * n_W)
        dA = np.ones((m, n_H, n_W, n_C)) * dA_filled
        return dA


class Flatten:
    def __init__(self):
        self.input_shape = None
        self.output = None

    def forward(self, data):
        self.input_shape = data.shape
        self.output = data.reshape(data.shape[0], -1)
        return self.output

    def backward(self, dZ):
        dA = dZ.reshape(self.input_shape)
        return dA


class Fully_connected:
    def __init__(self, shape_out=None, learning_rate=0.0003, epsilon=10 ** -8, Beta1=0.9, Beta2=0.999, decay_rate=0.95):
        self.output = None
        self.W = None
        self.B = None
        self.lr = learning_rate
        self.epsilon = epsilon
        self.Beta1 = Beta1
        self.Beta2 = Beta2
        self.decay_rate = decay_rate
        self.V_dw = 0
        self.V_db = 0
        self.S_dw = 0
        self.S_db = 0
        self.shape_out = shape_out
        self.Loss_value = None
        self.cache = None

    def softmax(self, Z):
        Z_normalized = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_normalized)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, data):
        if self.W is None:
            shape_in = data.shape[1]
            scale = np.sqrt(1.0 / shape_in)
            self.W = np.random.randn(shape_in, self.shape_out).astype(np.float32) * scale
            self.B = np.zeros((1, self.shape_out), dtype = np.float32)

        Z = np.dot(data, self.W) + self.B
        self.output = self.softmax(Z)
        self.cache = data

        return self.output

    def backward(self, dZ, num_of_mn_batch, epoch):
        data = self.cache
        dW = np.dot(data.T, dZ)
        dX = np.dot(dZ, self.W.T)
        dB = np.sum(dZ, axis=0, keepdims=True)
        lr = self.lr * (self.decay_rate ** epoch)

        (self.W, self.B, self.V_dw, self.V_db, self.S_dw, self.S_db) = Adam(self.W, self.B, dW, dB, self.V_dw,
                                                                            self.V_db,
                                                                            self.S_dw, self.S_db, self.Beta1,
                                                                            self.Beta2,
                                                                            num_of_mn_batch, lr)
        return dX


class Cross_entropy:
    def __init__(self):
        self.output = None
        self.cache = None

    def Loss_compute(self, Y_hat, label_set):
        m = Y_hat.shape[0]
        num_classes = Y_hat.shape[1]
        Y_onehot = np.zeros((m, num_classes))
        Y_onehot[np.arange(m), label_set] = 1
        Loss = np.sum(Y_hat * Y_onehot, axis=1)
        Cost = np.mean(-1 * np.log(Loss + 1e-05))
        self.output = Cost
        self.cache = (Y_hat, Y_onehot)
        return self.output

    def backward(self):
        Y_hat, Y_onehot = self.cache
        m = Y_hat.shape[0]
        dZ = (Y_hat - Y_onehot) / m
        return dZ


class ResNet50:
    def __init__(self):
        # Stage 1
        self.conv_layer = Conv_layer(filters=64, kernel_size=7, strides=2)
        self.BN_layer = BatchNorm()
        self.ReLU_layer = ReLU()
        self.max_pool_layer = Pooling_layer(kernel_size=3, strides=2, pooling_mode="max")
        # Stage 2
        self.conv_block_s2 = Conv_block(kernel_size=3, filters=[64, 64, 256], strides=1)
        self.ID_block_s2_1 = Identity_block(kernel_size=3, filters=[64, 64, 256])
        self.ID_block_s2_2 = Identity_block(kernel_size=3, filters=[64, 64, 256])
        # Stage 3
        self.conv_block_s3 = Conv_block(kernel_size=3, filters=[128, 128, 512], strides=2)
        self.ID_block_s3_1 = Identity_block(kernel_size=3, filters=[128, 128, 512])
        self.ID_block_s3_2 = Identity_block(kernel_size=3, filters=[128, 128, 512])
        self.ID_block_s3_3 = Identity_block(kernel_size=3, filters=[128, 128, 512])
        # Stage 4
        self.conv_block_s4 = Conv_block(kernel_size=3, filters=[256, 256, 1024], strides=2)
        self.ID_block_s4_1 = Identity_block(kernel_size=3, filters=[256, 256, 1024])
        self.ID_block_s4_2 = Identity_block(kernel_size=3, filters=[256, 256, 1024])
        self.ID_block_s4_3 = Identity_block(kernel_size=3, filters=[256, 256, 1024])
        self.ID_block_s4_4 = Identity_block(kernel_size=3, filters=[256, 256, 1024])
        self.ID_block_s4_5 = Identity_block(kernel_size=3, filters=[256, 256, 1024])
        # Stage 5
        self.conv_block_s5 = Conv_block(kernel_size=3, filters=[512, 512, 2048], strides=2)
        self.ID_block_s5_1 = Identity_block(kernel_size=3, filters=[512, 512, 2048])
        self.ID_block_s5_2 = Identity_block(kernel_size=3, filters=[512, 512, 2048])
        self.ID_block_s5_3 = Identity_block(kernel_size=3, filters=[512, 512, 2048])
        #  Final Stage
        self.avg_pool_layer = AVG_pooling_global()
        self.Flatten_layer = Flatten()
        self.FC1 = Fully_connected(128)
        self.ReLU4 = ReLU()
        self.FC2 = Fully_connected(32)
        self.ReLU5 = ReLU()
        self.FC3 = Fully_connected(4)
        self.output = None

    def zero_pad(self, X, pad):
        X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
        return X_pad

    def pre_processing(self, input):
        output = self.zero_pad(input, 3)
        return output

    def forward(self, X_input, training = False):
        # Stage 1
        X = self.pre_processing(X_input)
        X = self.conv_layer.forward(X)
        X = self.BN_layer.forward(X, training)
        X = self.ReLU_layer.forward(X)
        X = self.max_pool_layer.forward(X)
        # Stage 2
        X = self.conv_block_s2.forward(X, training)
        X = self.ID_block_s2_1.forward(X, training)
        X = self.ID_block_s2_2.forward(X, training)
        # Stage 3
        X = self.conv_block_s3.forward(X, training)
        X = self.ID_block_s3_1.forward(X, training)
        X = self.ID_block_s3_2.forward(X, training)
        X = self.ID_block_s3_3.forward(X, training)
        # Stage 4
        X = self.conv_block_s4.forward(X, training)
        X = self.ID_block_s4_1.forward(X, training)
        X = self.ID_block_s4_2.forward(X, training)
        X = self.ID_block_s4_3.forward(X, training)
        X = self.ID_block_s4_4.forward(X, training)
        X = self.ID_block_s4_5.forward(X, training)
        # Stage 5
        X = self.conv_block_s5.forward(X, training)
        X = self.ID_block_s5_1.forward(X, training)
        X = self.ID_block_s5_2.forward(X, training)
        X = self.ID_block_s5_3.forward(X, training)
        # Final Stage
        X = self.avg_pool_layer.forward(X)
        X = self.Flatten_layer.forward(X)
        X = self.FC1.forward(X)
        X = self.ReLU4.forward(X)

        X = self.FC2.forward(X)
        X = self.ReLU5.forward(X)

        X = self.FC3.forward(X)

        return X

    def Loss_compute(self, gt, pred):
        centre_pred_x = (pred[:,2] + pred[:,0]) / 2
        centre_pred_y = (pred[:,3] + pred[:,1]) / 2
        centre_gt_x = (gt[:,2] + gt[:,0]) / 2
        centre_gt_y = (gt[:,3] + gt[:,1]) /2

        rate_x = np.abs(centre_gt_x - centre_pred_x) / centre_gt_x
        rate_y = np.abs(centre_gt_y - centre_pred_y) / centre_gt_y
        rate = (rate_x / (rate_x + 0.2)) * (rate_y / ( rate_y + 0.2))
        rate = rate.reshape(-1,1)

        loss = np.mean(((gt - pred) ** 2) * (1 + rate))
        return loss

    def backward(self, dZ_input, num_of_mn_batch, epoch):
        dZ = self.FC3.backward(dZ_input, num_of_mn_batch, epoch)

        dZ = self.ReLU5.backward(dZ)
        dZ = self.FC2.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.ReLU4.backward(dZ)
        dZ = self.FC1.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.Flatten_layer.backward(dZ)
        dZ = self.avg_pool_layer.backward(dZ)

        dZ = self.ID_block_s5_3.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s5_2.backward(dZ, num_of_mn_batch,epoch)
        dZ = self.ID_block_s5_1.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.conv_block_s5.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s4_5.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s4_4.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s4_3.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s4_2.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s4_1.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.conv_block_s4.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s3_3.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s3_2.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s3_1.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.conv_block_s3.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s2_2.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.ID_block_s2_1.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.conv_block_s2.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.max_pool_layer.backward(dZ)
        dZ = self.ReLU_layer.backward(dZ)
        dZ = self.BN_layer.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.conv_layer.backward(dZ, num_of_mn_batch, epoch)

        return dZ

    def predict(self, input):
        result = self.forward(input)
        result = np.argmax(result, axis=1)
        return result

def save_model(model, filename):
    folder = os.path.dirname(filename)
    if folder and not os.path.dirname(filename):
        os.makedirs(folder)

    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saving model successfully!: {filename}")

    except Exception as e :
        print(f"Error while saving model!: {e}")

def load_model(filename):
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Loading model successfully!: {filename}")
        return model
    except Exception as e:
        print(f"Error while loading model!: {e}")
        return None

def convert_model_to_float32(model):
    print("⚙️ Đang chuyển đổi toàn bộ Model sang float32 để tăng tốc...")
    count = 0

    def recursive_convert(obj):
        nonlocal count
        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if isinstance(value, np.ndarray) and value.dtype == np.float64:
                    obj.__dict__[key] = value.astype(np.float32)
                    count += 1
                elif hasattr(value, '__dict__'):
                    recursive_convert(value)
                elif isinstance(value, list):
                    for item in value:
                        recursive_convert(item)

    recursive_convert(model)
    print(f"✅ Đã chuyển đổi {count} tham số (W, B, V, S...) sang float32.")

def reset_learning_rate(model, new_lr=0.0003):
    print(f"🚑 Đang khôi phục Learning Rate về {new_lr}...")
    count = 0

    def recursive_reset(obj):
        nonlocal count
        if hasattr(obj, 'lr'):
            obj.lr = new_lr
            count += 1

        if hasattr(obj, '__dict__'):
            for key, value in obj.__dict__.items():
                if hasattr(value, '__dict__'):
                    recursive_reset(value)
                elif isinstance(value, list):  # Phòng trường hợp layer nằm trong list
                    for item in value:
                        recursive_reset(item)

    recursive_reset(model)
    print(f"✅ Đã hồi phục Learning Rate cho {count} layers.")

def Visualization_result(pred, y_test_org):
    num_samples = len(pred)
    width_org = y_test_org[:,1].astype(np.float32)
    height_org = y_test_org[:,2].astype(np.float32)
    pred[:,0] *= width_org / 128
    pred[:,1] *= height_org / 128
    pred[:,2] *= width_org / 128
    pred[:,3] *= height_org / 128
    plt.figure(figsize=(20,20))
    img_paths = r'C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\Project\DL\Convolutional network\Obj_Detection\Raccoon_detection\Racoon_Images\images'
    for n,i in enumerate(range(num_samples)):
        xmin_pred = pred[i,0]
        ymin_pred = pred[i,1]
        xmax_pred = pred[i,2]
        ymax_pred = pred[i,3]
        xmin_gt = y_test_org[i,3]
        ymin_gt = y_test_org[i,4]
        xmax_gt = y_test_org[i,5]
        ymax_gt = y_test_org[i,6]

        x = np.sqrt(num_samples)
        x_rounded = np.round(x)
        if x_rounded < x:
            x = int(x_rounded)
            ax = plt.subplot(x, x+1, n+1)
        else:
            x= int(x_rounded)
            ax = plt.subplot(x, x, n + 1)
        img = cv2.imread(img_paths + '/' + y_test_org[i,0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        rect_pred = patches.Rectangle((xmin_pred, ymin_pred),xmax_pred - xmin_pred, ymax_pred - ymin_pred,
                                      linewidth=2, edgecolor='red', facecolor='none', label='Predicted')
        ax.add_patch(rect_pred)
        rect_gt = patches.Rectangle((xmin_gt, ymin_gt), xmax_gt - xmin_gt, ymax_gt - ymin_gt, linewidth=2,
                                    edgecolor='green', facecolor='none', label='Ground Truth')
        ax.add_patch(rect_gt)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return

def Train_model(model, Data_loader, epochs):
    print("Training model...")
    for epoch in range(epochs):
        running_loss = 0
        pbar = tqdm(Data_loader)
        number_of_mn_batch = len(Data_loader)
        total_img = 0
        for batch_idx, (X_loader, y_loader) in enumerate(pbar):
            X_numpy = np.asarray(X_loader.numpy()).astype(np.float32)
            Y_numpy = np.asarray(y_loader.numpy()).astype(np.float32)
            total_img += X_numpy.shape[0]

            y_hat = model.forward(X_numpy, training = True)
            loss = model.Loss_compute(Y_numpy, y_hat)

            running_loss += float(loss)
            dZ_initial = 2 * (y_hat - Y_numpy) / X_numpy.shape[0]

            global_step = batch_idx + number_of_mn_batch * epoch
            dZ = model.backward(dZ_initial,global_step, epoch)

            pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {float(loss):.4f}")

            del X_numpy, Y_numpy, y_hat, loss, dZ
            np.get_default_memory_pool().free_all_blocks()

        epoch_loss = running_loss / total_img
        print(f"\nKẾT THÚC EPOCH {epoch + 1}: Loss/Image: {epoch_loss:.4f}")
        save_name = "Parameter_cache.pkl"
        save_model(model, save_name)

def Valid_model(model, Data_loader, y_valid_org):
    print("Validating model...")
    running_loss = 0
    pbar = tqdm(Data_loader)
    total_img = 0
    pred = np.zeros((40, 4))
    idx = 0

    for batch_idx, (X_loader, y_loader) in enumerate(pbar):
        X_numpy = numpy.asarray(X_loader.numpy()).astype(numpy.float32)
        Y_numpy = numpy.asarray(y_loader.numpy()).astype(numpy.float32)
        total_img += X_numpy.shape[0]

        y_hat = model.forward(X_numpy)
        pred[idx:idx + y_hat.shape[0]] = y_hat
        idx += y_hat.shape[0]
        loss = model.Loss_compute(Y_numpy, y_hat)

        running_loss += loss

        del X_numpy, Y_numpy, y_hat, loss
        np.get_default_memory_pool().free_all_blocks()

    print(f'Total Loss: {float(running_loss):.4f}')
    loss = running_loss / total_img
    Visualization_result(pred, y_valid_org)
    return loss

def Test_model(model, Data_loader, y_test_org):
    print("Testing model...")
    running_loss = 0
    pbar = tqdm(Data_loader)
    total_img = 0
    pred = np.zeros((13,4))
    idx = 0
    for batch_idx, (X_loader, y_loader) in enumerate(pbar):
        X_numpy = np.asarray(X_loader.numpy()).astype(np.float32)
        Y_numpy = np.asarray(y_loader.numpy()).astype(np.float32)
        total_img += X_numpy.shape[0]

        y_hat = model.forward(X_numpy)
        pred[idx:idx + y_hat.shape[0]] = y_hat.get()
        idx += y_hat.shape[0]
        loss = model.Loss_compute(Y_numpy, y_hat)

        running_loss += loss

        del X_numpy, Y_numpy, y_hat, loss
        np.get_default_memory_pool().free_all_blocks()

    print(f'Total Loss: {float(running_loss):.4f}')
    loss = running_loss / total_img
    Visualization_result(pred, y_test_org)
    return loss

def Activate_model(model, type):
    X_train, X_valid, X_test, y_train, y_valid, y_test, y_test_org, y_valid_org = PreProcess()

    train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size= len(X_train)).batch(8).prefetch(tf.data.AUTOTUNE)
    valid_loader = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(8).prefetch(tf.data.AUTOTUNE)
    test_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(8).prefetch(tf.data.AUTOTUNE)
    if type == "train":
        reset_learning_rate(model, new_lr = 0.0001 * 1 ** 0)
        Train_model(model, train_loader, epochs = 5)
        print(f"Training process is completed !!!")
    elif type == "valid":
        loss = Valid_model(model, valid_loader, y_valid_org)
        print(f"Validating process is completed - Avg Loss = {loss:.4f} !!!")
    elif type == "test":
        loss = Test_model(model, test_loader, y_test_org)
        print(f"Testing process is completed - Avg Loss = {loss:.4f} !!!")

if __name__ == '__main__':
    model_path = 'Parameter_cache.pkl'
    type = "valid"
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Loading available model {model_path} successfully ")
    else:
        model = ResNet50()
        print(f"Initialized new model: {model_path} ")

    try:
        Activate_model(model, type)
    except KeyboardInterrupt:
        if type == "train":
            print("\nĐã dừng training thủ công. Đang lưu khẩn cấp...")
            save_model(model, 'Temp_Parameter.pkl')
        else:
            print("\nĐã dừng quá trình validation/test thủ công")
    except Exception as e:
        print(f"\n[LỖI NGHIÊM TRỌNG]: Chương trình crash vì lỗi sau:")
        traceback.print_exc()
