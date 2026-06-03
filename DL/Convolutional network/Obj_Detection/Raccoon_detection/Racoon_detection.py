import os
import pickle
import traceback
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from tqdm import tqdm
import cv2
import tensorflow as tf

def PreProcess():
    path_imgs = r'C:\Users\MSI LAPTOP\Downloads\Documents\CODE\ML\PycharmPractice\Project\DL\Convolutional network\Obj_Detection\Raccoon_detection\Racoon_Images\images'
    labels = pd.read_csv('train_labels_.csv')
    new_size = 128
    X_train = []
    X_valid = []
    X_test = []
    y_test_org = np.asarray(labels[['filename','width','height','xmin','ymin','xmax','ymax']].iloc[160:].values)
    y_valid_org = np.asarray(labels[['filename','width','height','xmin','ymin','xmax','ymax']].iloc[120:160].values)

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

    X_train = np.array(X_train)
    X_train = X_train / 255
    X_valid = np.array(X_valid)
    X_valid = X_valid / 255
    X_test = np.array(X_test)
    X_test = X_test / 255
    y_train = np.asarray(labels[cols].iloc[:120].values).astype(np.float32)
    y_valid = np.asarray(labels[cols].iloc[120:160].values).astype(np.float32)
    y_test = np.asarray(labels[cols].iloc[160:].values).astype(np.float32)

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

    W = W - lr * V_dw_corr / (cp.sqrt(S_dw_corr) + 1e-8)
    B = B - lr * V_db_corr / (cp.sqrt(S_db_corr) + 1e-8)

    return W, B, V_dw, V_db, S_dw, S_db

class Conv_layer:
    # Initialize
    def __init__(self, filters=None, kernel_size=3, pad=0, strides=1, padding_mode=None, learning_rate=0.0011,
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
        X_pad = cp.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
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

        A_col_view = cp.lib.stride_tricks.as_strided(A_prev_pad, shape=new_shape, strides=new_strides)
        A_col = A_col_view.reshape(m * n_H * n_W, -1)
        W_col = self.W.reshape(-1, n_filters)

        Z_col = cp.dot(A_col, W_col) + self.B.reshape(1, n_filters)
        Z = Z_col.reshape(m, n_H, n_W, n_filters)

        cache = (A_prev, self.W, self.B, pad, strides)

        return Z, cache

    def pre_process(self, data):
        if self.W is None:
            fan_in = self.kernel_size * self.kernel_size * data.shape[3]
            scale = cp.sqrt(2.0 / fan_in)
            self.W = cp.random.randn(self.kernel_size, self.kernel_size, data.shape[3], self.filters).astype(cp.float32) * scale
            self.B = cp.zeros((1, 1, 1, self.filters), dtype = cp.float32)

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

        dB = cp.sum(dZ, axis=(0, 1, 2)).reshape(1, 1, 1, n_filters)

        A_prev_pad = self.zero_pad(A_prev, pad)
        s0, s1, s2, s3 = A_prev_pad.strides
        new_shape = (m, n_H, n_W, f, f, pre_channels)
        new_strides = (s0, s1 * strides, s2 * strides, s1, s2, s3)

        A_col_view = cp.lib.stride_tricks.as_strided(A_prev_pad, shape=new_shape, strides=new_strides)
        A_col = A_col_view.reshape(m * n_H * n_W, -1)
        dZ_col = dZ.reshape(-1, n_filters)

        dW_col = cp.dot(A_col.T, dZ_col)
        dW = dW_col.reshape(f, f, pre_channels, n_filters)

        W_reshape = W.reshape(-1, n_filters)
        dA_col = cp.dot(dZ_col, W_reshape.T)
        dA_col_reshaped = dA_col.reshape(m, n_H, n_W, f, f, pre_channels)

        dA_prev_pad = cp.zeros_like(A_prev_pad)

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

        A_prev_view = cp.lib.stride_tricks.as_strided(
            A_prev, shape=new_shape, strides=new_strides
        )

        if self.pooling_mode == "max":
            A = cp.max(A_prev_view, axis=(3, 4))
        elif self.pooling_mode == "average":
            A = cp.mean(A_prev_view, axis=(3, 4))
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
        dA_prev = cp.zeros(A_prev.shape)

        if self.pooling_mode == "max":
            s0, s1, s2, s3 = A_prev.strides
            new_shape = (m, n_H, n_W, f, f, n_C)
            new_strides = (s0, s1 * strides, s2 * strides, s1, s2, s3)
            A_prev_windows = cp.lib.stride_tricks.as_strided(A_prev, shape=new_shape, strides=new_strides)

            max_val = cp.max(A_prev_windows, axis=(3, 4), keepdims=True)
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
    def __init__(self, eps=1e-05, gamma=1, beta=0, mean=None, var=None, learning_rate = 0.0011,
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
        if training:
            self.mean = cp.mean(data, axis=(0, 1, 2), keepdims=True)
            self.var = cp.var(data, axis=(0, 1, 2), keepdims=True)
        x_hat = (data - self.mean) / (cp.sqrt(self.var + self.eps))
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
        dBeta = cp.sum(dZ, axis=(0, 1, 2), keepdims=True)
        dGamma = cp.sum(dZ * X_hat, axis=(0, 1, 2), keepdims=True)
        dVar = cp.sum(dX_hat * bias * -1 / 2 * cp.power(temp, -3 / 2), axis=(0, 1, 2), keepdims=True)
        dMean = cp.sum(dX_hat * -1 / cp.sqrt(temp) + dVar * -2 * bias / m, axis=(0, 1, 2), keepdims=True)
        dX = dVar * 2 * bias / m + dMean / m + dX_hat / cp.sqrt(temp)

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
        self.cache = cp.copy(data)
        data[data < self.threshold] = data[data < self.threshold] * 0.01
        return data

    def forward(self, data):
        self.output = self.ReLU(data)
        return self.output

    def backward(self, dZ):
        X = self.cache
        dA = cp.copy(dZ)
        dA[X <= self.threshold] = dA[X <= self.threshold] * 0.01
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
    def __init__(self, shape_out=None, learning_rate = 0.0011, epsilon=10 ** -8, Beta1=0.9, Beta2=0.999, decay_rate=0.95):
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

    def forward(self, data):
        if self.W is None:
            shape_in = data.shape[1]
            scale = cp.sqrt(1.0 / shape_in)
            self.W = cp.random.randn(shape_in, self.shape_out).astype(cp.float32) * scale
            self.B = cp.zeros((1, self.shape_out), dtype = cp.float32)

        Z = cp.dot(data, self.W) + self.B
        self.output = Z
        self.cache = data

        return self.output

    def backward(self, dZ, num_of_mn_batch, epoch):
        data = self.cache
        dW = cp.dot(data.T, dZ)
        dX = cp.dot(dZ, self.W.T)
        dB = cp.sum(dZ, axis=0, keepdims=True)
        lr = self.lr * (self.decay_rate ** epoch)

        (self.W, self.B, self.V_dw, self.V_db, self.S_dw, self.S_db) = Adam(self.W, self.B, dW, dB, self.V_dw,
                                                                            self.V_db,
                                                                            self.S_dw, self.S_db, self.Beta1,
                                                                            self.Beta2,
                                                                            num_of_mn_batch, lr)
        return dX


class Model:
    def __init__(self):
        self.cache = None

        self.Conv1 = Conv_layer(64,3)
        self.Bn1 = BatchNorm()
        self.ReLU1 = ReLU()
        self.Mx_pooling1 = Pooling_layer(3,3,"max")
        self.Conv2 = Conv_layer(128,3)
        self.BN2 = BatchNorm()
        self.ReLU2 = ReLU()
        self.Mx_pooling2 = Pooling_layer(3,3,"max")
        self.Conv3 = Conv_layer(256,3)
        self.BN3 = BatchNorm()
        self.ReLU3 = ReLU()
        self.Mx_pooling3 = Pooling_layer(3,3,"max")
        self.Flatten = Flatten()
        self.FC1 = Fully_connected(128)
        self.ReLU4 = ReLU()
        self.FC2 = Fully_connected(32)
        self.ReLU5 = ReLU()
        self.FC3 = Fully_connected(4)

    def forward(self, data, training = False):
        a = self.Conv1.forward(data)
        a = self.Bn1.forward(a, training)
        a = self.ReLU1.forward(a)
        a = self.Mx_pooling1.forward(a)

        a = self.Conv2.forward(a)
        a = self.BN2.forward(a, training)
        a = self.ReLU2.forward(a)
        a = self.Mx_pooling2.forward(a)

        a = self.Conv3.forward(a)
        a = self.BN3.forward(a, training)
        a = self.ReLU3.forward(a)
        a = self.Mx_pooling3.forward(a)

        a = self.Flatten.forward(a)

        a = self.FC1.forward(a)
        a = self.ReLU4.forward(a)

        a = self.FC2.forward(a)
        a = self.ReLU5.forward(a)

        a = self.FC3.forward(a)

        return a

    def Loss_compute(self, gt, pred):
        centre_pred_x = (pred[:,2] + pred[:,0]) / 2
        centre_pred_y = (pred[:,3] + pred[:,1]) / 2
        centre_gt_x = (gt[:,2] + gt[:,0]) / 2
        centre_gt_y = (gt[:,3] + gt[:,1]) /2

        rate_x = np.abs(centre_gt_x - centre_pred_x) / centre_gt_x
        rate_y = np.abs(centre_gt_y - centre_pred_y) / centre_gt_y
        rate = (rate_x / (rate_x + 0.2)) * (rate_y / ( rate_y + 0.2))
        rate = rate.reshape(-1,1)

        loss = cp.mean(((gt - pred) ** 2) * (1 + rate))
        return loss

    def backward(self, dZ, num_of_mn_batch, epoch):
        dZ = self.FC3.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.ReLU5.backward(dZ)
        dZ = self.FC2.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.ReLU4.backward(dZ)
        dZ = self.FC1.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.Flatten.backward(dZ)

        dZ = self.Mx_pooling3.backward(dZ)
        dZ = self.ReLU3.backward(dZ)
        dZ = self.BN3.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.Conv3.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.Mx_pooling2.backward(dZ)
        dZ = self.ReLU2.backward(dZ)
        dZ = self.BN2.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.Conv2.backward(dZ, num_of_mn_batch, epoch)

        dZ = self.Mx_pooling1.backward(dZ)
        dZ = self.ReLU1.backward(dZ)
        dZ = self.Bn1.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.Conv1.backward(dZ, num_of_mn_batch, epoch)

        return dZ

    def clear_all_caches(self):
        for layer in [self.Conv1, self.Bn1, self.ReLU1, self.Mx_pooling1,
                      self.Conv2, self.BN2, ...]:
            if hasattr(layer, 'cache'):
                layer.cache = None
            if hasattr(layer, 'output'):
                layer.output = None

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f'Loaded model from {model_path} successfully')
        return model
    except Exception as e:
        print(f"error loading model: {e}")
        return None

def save_model(model, filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder) :
        os.makedirs(folder)

    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saving model to {filename} successfully")
    except Exception as e:
        print(f"Error while saving model: {e}")

def restore_lr(model, new_lr):
    print("Restoring learning rate...")
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
                elif isinstance(value, list):
                    for item in value:
                        recursive_reset(item)
    recursive_reset(model)
    print(f"Restored learning rate = {new_lr} for {count} layers successfully !!!")

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
            X_numpy = cp.asarray(X_loader.numpy()).astype(cp.float32)
            Y_numpy = cp.asarray(y_loader.numpy()).astype(cp.float32)
            total_img += X_numpy.shape[0]

            y_hat = model.forward(X_numpy, training = True)
            loss = model.Loss_compute(Y_numpy, y_hat)

            running_loss += float(loss)
            dZ_initial = 2 * (y_hat - Y_numpy) / X_numpy.shape[0]

            global_step = batch_idx + number_of_mn_batch * epoch
            dZ = model.backward(dZ_initial,global_step, epoch)
            model.clear_all_caches()

            pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Loss: {float(loss):.4f}")

            del X_numpy, Y_numpy, y_hat, loss, dZ
            cp.get_default_memory_pool().free_all_blocks()

        loss = running_loss / total_img
        print(f"\nKẾT THÚC EPOCH {epoch }: Loss/Image: {loss:.4f}")
        save_name = "Parameter_cache.pkl"
        save_model(model, save_name)
        return loss

def Valid_model(model, Data_loader, y_valid_org, plot = True):
    print("Validating model...")
    running_loss = 0
    pbar = tqdm(Data_loader)
    total_img = 0
    pred = np.zeros((40, 4))
    idx = 0

    for batch_idx, (X_loader, y_loader) in enumerate(pbar):
        X_numpy = cp.asarray(X_loader.numpy()).astype(cp.float32)
        Y_numpy = cp.asarray(y_loader.numpy()).astype(cp.float32)
        total_img += X_numpy.shape[0]

        y_hat = model.forward(X_numpy)
        pred[idx:idx + y_hat.shape[0]] = y_hat.get()
        idx += y_hat.shape[0]
        loss = model.Loss_compute(Y_numpy, y_hat)

        running_loss += loss

        del X_numpy, Y_numpy, y_hat, loss
        cp.get_default_memory_pool().free_all_blocks()

    print(f'Total Loss: {float(running_loss):.4f}')
    loss = running_loss / total_img
    if plot == True:
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
        X_numpy = cp.asarray(X_loader.numpy()).astype(cp.float32)
        Y_numpy = cp.asarray(y_loader.numpy()).astype(cp.float32)
        total_img += X_numpy.shape[0]

        y_hat = model.forward(X_numpy)
        pred[idx:idx + y_hat.shape[0]] = y_hat.get()
        idx += y_hat.shape[0]
        loss = model.Loss_compute(Y_numpy, y_hat)

        running_loss += loss

        del X_numpy, Y_numpy, y_hat, loss
        cp.get_default_memory_pool().free_all_blocks()

    print(f'Total Loss: {float(running_loss):.4f}')
    loss = running_loss / total_img
    Visualization_result(pred, y_test_org)
    return loss

def Model_loss_tracking(model, train_loader, valid_loader, y_valid_org, nums_of_epochs):
    training_loss = []
    validating_loss = []

    for epoch in range(nums_of_epochs):
        training_loss.append(Train_model(model, train_loader, epochs=1))
        validating_loss.append(Valid_model(model, valid_loader, y_valid_org, plot=False))
        print(f'Complete epoch {epoch} / {nums_of_epochs} !!! ')
    print(len(training_loss))
    print(len(validating_loss))

    plt.figure(figsize=(10, 6))
    plt.plot(cp.array(training_loss).get(), label='Training Loss', color='blue', linewidth=2)
    plt.plot(cp.array(validating_loss).get(), label='Validation Loss', color='red', linewidth=2)

    plt.title('Model Loss Tracking', fontsize=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def Activate_model(model, type):
    X_train, X_valid, X_test, y_train, y_valid, y_test, y_test_org, y_valid_org = PreProcess()

    train_loader = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size= len(X_train)).batch(8).prefetch(tf.data.AUTOTUNE)
    valid_loader = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(8).prefetch(tf.data.AUTOTUNE)
    test_loader = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(8).prefetch(tf.data.AUTOTUNE)
    if type == "train":
        restore_lr(model, new_lr = 0.0011 * 0.97 ** 0)
        Train_model(model, train_loader, epochs = 0)
        print(f"Training process is completed !!!")
    elif type == "valid":
        loss = Valid_model(model, valid_loader, y_valid_org)
        print(f"Validating process is completed - Avg Loss = {loss:.4f} !!!")
    elif type == "test":
        loss = Test_model(model, test_loader, y_test_org)
        print(f"Testing process is completed - Avg Loss = {loss:.4f} !!!")
    elif type == 'model_loss_tracking':
        Model_loss_tracking(model, train_loader, valid_loader, y_valid_org, nums_of_epochs = 30)
        print(f"Model loss tracking process is completed !!!")

if __name__ == '__main__':
    model_path = 'Parameter_cache.pkl'
    type = "test"
    if os.path.exists(model_path):
        model = load_model(model_path)
        if model is not None:
            print(f"Loading available model {model_path} successfully ")
        else:
            model = Model()
            print(f"Initialized new model successfully!!!")
    else:
        model = Model()
        print(f"Initialized new model: {model_path} successfully !!!")

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
