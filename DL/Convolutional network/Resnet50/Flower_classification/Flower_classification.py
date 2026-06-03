import cv2
import torch
import cupy as np
import pandas as pd
from matplotlib import pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import pickle
import os
from tqdm import tqdm


try:
    print(f"CuPy đang chạy trên thiết bị: {np.cuda.Device(0).use()}")
    gpu_mode = True
except Exception as e:
    print("CẢNH BÁO: Không tìm thấy GPU hoặc chưa cài CuPy đúng cách. Đang chạy fallback...")
    gpu_mode = False

data_dir = 'flower_photos/'
train_dir = data_dir + 'train/'
valid_dir = data_dir + 'validation/'
test_dir = data_dir + 'test/'

batch_size = 8

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
valid_data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
test_data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

train_dataset = datasets.ImageFolder(train_dir, transform=train_data_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_data_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_data_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4)


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


class Leaky_ReLU():
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
        self.ReLU_layer1 = Leaky_ReLU()
        self.conv_layer2 = Conv_layer(self.filters[1], self.kernel_size, padding_mode="same")
        self.BN_layer2 = BatchNorm()
        self.ReLU_layer2 = Leaky_ReLU()
        self.conv_layer3 = Conv_layer(self.filters[2], 1, padding_mode="same")
        self.BN_layer3 = BatchNorm()
        self.ReLU_layer3 = Leaky_ReLU()

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
        self.ReLU_layer1 = Leaky_ReLU()
        self.conv_layer2 = Conv_layer(self.filters[1], self.kernel_size, padding_mode="same")
        self.BN_layer2 = BatchNorm()
        self.ReLU_layer2 = Leaky_ReLU()
        self.conv_layer3 = Conv_layer(self.filters[2], 1, padding_mode="valid")
        self.BN_layer3 = BatchNorm()
        self.ReLU_layer3 = Leaky_ReLU()
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
        self.ReLU_layer = Leaky_ReLU()
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
        self.FC_layer = Fully_connected(5)
        self.Classifier = Cross_entropy()
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
        X = self.FC_layer.forward(X)

        return X

    def Loss_compute(self, Y_hat, label_set):
        dZ = self.Classifier.Loss_compute(Y_hat, label_set)
        return dZ

    def backward(self, dZ_input, num_of_mn_batch, epoch):
        dZ = self.FC_layer.backward(dZ_input, num_of_mn_batch, epoch)
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

def Visualization_result(pred):
    main_path = r"C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\Project\DL\Convolutional network\Resnet50\Flower_classification\flower_photos\test"
    img_paths = []
    labels = []
    num_samples = pred.shape[0]
    for root, dir, files in os.walk(main_path):
        for file in files:
            img_paths.append(os.path.join(root, file))
            labels.append(os.path.basename(root))

    for idx, img_path in enumerate(img_paths):
        x = np.sqrt(num_samples)
        x_rounded = np.round(x)
        if x_rounded < x:
            x = int(x_rounded)
            ax = plt.subplot(x, x + 1, idx + 1)
        else:
            x = int(x_rounded)
            ax = plt.subplot(x, x, idx + 1)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.axis('off')
        if pred[idx] == 0:
            predicted_label = f'daisy/{labels[idx]}'
        elif pred[idx] == 1:
            predicted_label = f'dandelion/{labels[idx]}'
        elif pred[idx] == 2:
            predicted_label = f'roses/{labels[idx]}'
        elif pred[idx] == 3:
            predicted_label = f'sunflowers/{labels[idx]}'
        elif pred[idx] == 4:
            predicted_label = f'tulips/{labels[idx]}'

        ax.set_title(predicted_label)
        ax.imshow(img)

    plt.tight_layout()
    plt.show()
    return

def Train_model(model, epochs):
    print("Bắt đầu training trên CUDA (Nếu đã cài CuPy)...")
    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        pbar = tqdm(train_loader)
        number_of_batch = len(train_loader)
        for batch_idx, (data, target) in enumerate(pbar):
            data: torch.Tensor
            target: torch.Tensor
            X_numpy = np.asarray(data.permute(0, 2, 3, 1).numpy()).astype(np.float32)
            Y_numpy = np.asarray(target.numpy())

            Y_hat = model.forward(X_numpy, training = True)
            loss = model.Loss_compute(Y_hat, Y_numpy)

            running_loss += float(loss)

            predictions = np.argmax(Y_hat, axis=1)
            correct_predictions += int(np.sum(predictions == Y_numpy))
            total_samples += X_numpy.shape[0]

            dZ_initial = model.Classifier.backward()
            global_step = batch_idx + number_of_batch * epoch
            model.backward(dZ_initial, global_step, epoch)
            GD = model.conv_layer.V_dw
            print(np.mean(GD))
            pbar.set_description(
                f"Epoch {epoch + 1}/{epochs} - Loss: {float(loss):.4f} - Acc: {correct_predictions / total_samples:.2%}")

            del X_numpy, Y_numpy, Y_hat, loss, predictions, dZ_initial

            if batch_idx % 20 == 0:
                np.get_default_memory_pool().free_all_blocks()

        epoch_loss = running_loss / number_of_batch
        epoch_acc = correct_predictions / total_samples
        print(f"\nKẾT THÚC EPOCH {epoch + 1}: Avg Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2%}\n")
        # save_name = "Para_cache.pkl"
        # save_model(model, save_name)

def Valid_model(model):
    print("Bắt đầu Validation process trên CUDA (Nếu đã cài CuPy)...")
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    pbar = tqdm(valid_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data: torch.Tensor
        target: torch.Tensor
        X_numpy = np.asarray(data.permute(0, 2, 3, 1).numpy()).astype(np.float32)
        Y_numpy = np.asarray(target.numpy())

        Y_hat = model.forward(X_numpy)
        loss = model.Loss_compute(Y_hat, Y_numpy)

        running_loss += float(loss)
        predictions = np.argmax(Y_hat, axis=1)
        correct_predictions += int(np.sum(predictions == Y_numpy))

        total_samples += X_numpy.shape[0]

        pbar.set_description(
            f"Loss: {float(loss):.4f} - Acc: {correct_predictions / total_samples:.2%}")

        del X_numpy, Y_numpy, loss, predictions, Y_hat
        np.get_default_memory_pool().free_all_blocks()

    loss = running_loss / len(valid_loader)
    accurating = correct_predictions / total_samples
    print(f"Completed Validation process !!!- Loss: {float(loss):.4f} - Cost = {float(running_loss):.4f} - Acc: {accurating:.2%}")
    return accurating

def Test_model(model):
    print("Bắt đầu Testing process trên CUDA (Nếu đã cài CuPy)...")
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    pbar = tqdm(test_loader)
    pred = np.zeros(len(test_dataset))
    for batch_idx, (data, target) in enumerate(pbar):
        X_numpy: torch.Tensor
        Y_numpy: torch.Tensor

        X_numpy = np.asarray(data.permute(0, 2, 3, 1).numpy()).astype(np.float32)
        Y_numpy = np.asarray(target.numpy())

        Y_hat = model.forward(X_numpy)
        loss = model.Loss_compute(Y_hat, Y_numpy)

        running_loss += float(loss)
        predictions = np.argmax(Y_hat, axis=1)
        pred[batch_idx * 8 : (batch_idx + 1) * 8] = predictions

        correct_predictions += int(np.sum(predictions == Y_numpy))
        for i in range(predictions.shape[0]):
            print(f'{predictions[i]:.2f} | {Y_numpy[i]:.2f}')

        total_samples += X_numpy.shape[0]
        print(f'result: {correct_predictions}/{total_samples}')

        pbar.set_description(
            f"Loss: {float(loss):.4f} - Acc: {correct_predictions / total_samples:.2%}")
        del X_numpy, Y_numpy, loss, predictions, Y_hat
        np.get_default_memory_pool().free_all_blocks()

    loss = running_loss / len(test_loader)
    accurating = correct_predictions / total_samples
    print(f"Completed Testing process !!!- Loss: {float(loss):.4f} - Cost = {float(running_loss):.4f} - Acc: {accurating:.2%}")
    Visualization_result(pred)
    return accurating

def activate_model(model, type, model_name):
    print("Bắt đầu chạy model trên CUDA (Nếu đã cài CuPy)...")
    if type == "train":
        # convert_model_to_float32(resnet_model)
        reset_learning_rate(resnet_model, new_lr=0.0003 * 0.95 ** 43)
        Train_model(model, epochs = 2)
        # print("Bắt đầu Overfitting rồi nên ko train nữa")
    elif type == "valid":
        accuracy = Valid_model(model)
        print(f"Validating accuracy of model {model_name}: {accuracy:.2%}")
    elif type == "test":
        accuracy = Test_model(model)
        print(f"Testing accuracy of model {model_name}: {accuracy:.2%}")
    else:
        print("Sai kiểu activate model !")
    return

if __name__ == '__main__':
    model_path = 'AccTest_=76%.pkl'
    type = "test"
    if os.path.exists(model_path):
        print("Đang tải lại model cũ...")
        resnet_model = load_model(model_path)
    else:
        print("Tạo model mới hoàn toàn...")
        resnet_model = ResNet50()
        # print("Bắt đầu Overfitting rồi nên ko train nữa")

    try:
        activate_model(resnet_model, type, model_path)
    except KeyboardInterrupt:
        if type == "train":
            print("Đã dừng training thủ công. Đang lưu khẩn cấp...")
            save_model(resnet_model, 'Temp_Parameter.pkl')
        else:
            print("Đã dừng quá trình validation/test thủ công")

