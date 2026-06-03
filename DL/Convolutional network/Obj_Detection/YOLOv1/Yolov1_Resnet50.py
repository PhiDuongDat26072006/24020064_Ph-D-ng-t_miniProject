import sys

import cupy as cp
from utils import Adam

class Conv_layer:
    # Initialize
    def __init__(self, filters=None, kernel_size=3, pad=0, strides=1, padding_mode=None, learning_rate=0.00001,
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

        A_prev = data

        return A_prev

    def forward(self, input):
        A_prev = self.pre_process(input)
        self.output, self.cache = self.conv_forward(A_prev, self.pad, self.strides)
        return self.output

    # Backward
    def backward(self, dZ, num_of_mn_batch, epoch):
        (A_prev, W, B, pad, strides) = self.cache
        (m, n_H, n_W, n_C) = dZ.shape
        (f, f, pre_channels, n_filters) = W.shape

        dB = cp.sum(dZ, axis=(0, 1, 2)).reshape(1, 1, 1, n_filters)

        A_prev_pad = self.zero_pad(A_prev, pad)
        pH, pW = A_prev_pad.shape[1], A_prev_pad.shape[2]

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

        # ✅ Vectorized col2im — không có Python loop
        row_idx = cp.arange(f)[:, None] + cp.arange(n_H)[None, :] * strides  # (f, n_H)
        col_idx = cp.arange(f)[:, None] + cp.arange(n_W)[None, :] * strides  # (f, n_W)
        flat_idx = (row_idx[:, :, None, None] * pW + col_idx[None, None, :, :]).reshape(-1)  # (f*n_H*f*n_W,)

        # (m, n_H, n_W, f, f, C) → (m, f, n_H, f, n_W, C) → (m, f*n_H*f*n_W, C)
        d = dA_col_reshaped.transpose(0, 3, 1, 4, 2, 5).reshape(m, -1, pre_channels)

        batch_offset = (cp.arange(m) * (pH * pW)).reshape(m, 1)
        global_idx = (batch_offset + flat_idx[None, :]).reshape(-1)

        dA_flat = cp.zeros((m * pH * pW, pre_channels), dtype=dA_col_reshaped.dtype)
        cp.add.at(dA_flat, global_idx, d.reshape(-1, pre_channels))
        dA_prev_pad = dA_flat.reshape(m, pH, pW, pre_channels)

        if pad != 0:
            dA_prev = dA_prev_pad[:, pad:-pad, pad:-pad, :]
        else:
            dA_prev = dA_prev_pad

        lr = self.lr * (self.decay_rate ** epoch)
        (self.W, self.B, self.V_dw, self.V_db, self.S_dw, self.S_db) = Adam(self.W, self.B, dW, dB, self.V_dw,
                                                                            self.V_db, self.S_dw, self.S_db,
                                                                            self.Beta1, self.Beta2,
                                                                            num_of_mn_batch, lr)
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
        pH, pW = A_prev.shape[1], A_prev.shape[2]

        dA_prev = cp.zeros(A_prev.shape, dtype=A_prev.dtype)

        if self.pooling_mode == "max":
            s0, s1, s2, s3 = A_prev.strides
            new_shape = (m, n_H, n_W, f, f, n_C)
            new_strides = (s0, s1 * strides, s2 * strides, s1, s2, s3)

            A_prev_windows = cp.lib.stride_tricks.as_strided(A_prev, shape=new_shape, strides=new_strides)

            max_val = cp.max(A_prev_windows, axis=(3, 4), keepdims=True)
            mask = (A_prev_windows == max_val)

            dA_expanded = dA.reshape(m, n_H, n_W, 1, 1, n_C)
            d_window = mask * dA_expanded  # (m, n_H, n_W, f, f, n_C)

            # ✅ Vectorized col2im — không có Python loop
            row_idx = cp.arange(f)[:, None] + cp.arange(n_H)[None, :] * strides  # (f, n_H)
            col_idx = cp.arange(f)[:, None] + cp.arange(n_W)[None, :] * strides  # (f, n_W)
            flat_idx = (row_idx[:, :, None, None] * pW + col_idx[None, None, :, :]).reshape(-1)

            # (m, n_H, n_W, f, f, C) → (m, f, n_H, f, n_W, C) → (m, f*n_H*f*n_W, C)
            d = d_window.transpose(0, 3, 1, 4, 2, 5).reshape(m, -1, n_C)

            batch_offset = (cp.arange(m) * (pH * pW)).reshape(m, 1)
            global_idx = (batch_offset + flat_idx[None, :]).reshape(-1)

            dA_flat = cp.zeros((m * pH * pW, n_C), dtype=d_window.dtype)
            cp.add.at(dA_flat, global_idx, d.reshape(-1, n_C))
            dA_prev = dA_flat.reshape(m, pH, pW, n_C)

        elif self.pooling_mode == "average":
            da = dA / (f * f)

            # ✅ Vectorized — không có Python loop
            row_idx = cp.arange(f)[:, None] + cp.arange(n_H)[None, :] * strides  # (f, n_H)
            col_idx = cp.arange(f)[:, None] + cp.arange(n_W)[None, :] * strides  # (f, n_W)
            flat_idx = (row_idx[:, :, None, None] * pW + col_idx[None, None, :, :]).reshape(-1)

            # Broadcast da sang (m, f*n_H*f*n_W, C)
            d = cp.broadcast_to(da[:, None, :, :, :], (m, 1, n_H, n_W, n_C))
            d = cp.broadcast_to(d, (m, f * f, n_H, n_W, n_C)).reshape(m, -1, n_C)

            batch_offset = (cp.arange(m) * (pH * pW)).reshape(m, 1)
            global_idx = (batch_offset + flat_idx[None, :]).reshape(-1)

            dA_flat = cp.zeros((m * pH * pW, n_C), dtype=da.dtype)
            cp.add.at(dA_flat, global_idx, d.reshape(-1, n_C))
            dA_prev = dA_flat.reshape(m, pH, pW, n_C)

        return dA_prev


class BatchNorm:
    # Initialize
    def __init__(self, eps=1e-05, gamma=1, beta=0, mean=None, var=None, learning_rate=0.00001,
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
            self.mean = cp.mean(data, axis=(0, 1, 2), keepdims=True)
            self.var = cp.var(data, axis=(0, 1, 2), keepdims=True)

        x_hat = (data - self.mean) / (cp.sqrt(self.var + self.eps))
        y = self.gamma * x_hat + self.beta
        self.cache = x_hat

        return y

    # Backward
    def backward(self, dZ, num_of_mn_batch, epoch):
        X_hat = self.cache
        X = X_hat * (cp.sqrt(self.var + self.eps)) + self.mean

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


class LeakyReLU():
    def __init__(self):
        self.threshold = 0.0
        self.cache = None

    def forward(self, data):
        self.cache = cp.copy(data)
        output = cp.where(data < self.threshold, data * 0.01, data)
        return output

    def backward(self, dZ):
        X = self.cache
        dA = cp.copy(dZ)
        dA[X < self.threshold] *= 0.01
        return dA


class Identity_block:
    def __init__(self, kernel_size=3, filters=None):
        self.kernel_size = kernel_size
        self.filters = filters
        self.conv_layer1 = Conv_layer(self.filters[0], 1, padding_mode="same")
        self.BN_layer1 = BatchNorm()
        self.ReLU_layer1 = LeakyReLU()
        self.conv_layer2 = Conv_layer(self.filters[1], self.kernel_size, padding_mode="same")
        self.BN_layer2 = BatchNorm()
        self.ReLU_layer2 = LeakyReLU()
        self.conv_layer3 = Conv_layer(self.filters[2], 1, padding_mode="same")
        self.BN_layer3 = BatchNorm()
        self.ReLU_layer3 = LeakyReLU()

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
        self.ReLU_layer1 = LeakyReLU()
        self.conv_layer2 = Conv_layer(self.filters[1], self.kernel_size, padding_mode="same")
        self.BN_layer2 = BatchNorm()
        self.ReLU_layer2 = LeakyReLU()
        self.conv_layer3 = Conv_layer(self.filters[2], 1, padding_mode="valid")
        self.BN_layer3 = BatchNorm()
        self.ReLU_layer3 = LeakyReLU()
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
        self.output = cp.mean(data, axis=(1, 2), keepdims=True)
        self.cache = data.shape
        return self.output

    def backward(self, dZ):
        (m, n_H, n_W, n_C) = self.cache
        dA_filled = dZ / (n_H * n_W)
        dA = cp.ones((m, n_H, n_W, n_C)) * dA_filled
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
    def __init__(self, shape_out=None, learning_rate=0.00001, epsilon=10 ** -8, Beta1=0.9, Beta2=0.999, decay_rate=0.95):
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
        Z_normalized = Z - cp.max(Z, axis=3, keepdims=True)
        exp_Z = cp.exp(Z_normalized)
        return exp_Z / cp.sum(exp_Z, axis=3, keepdims=True)

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

class ResNet50:
    def __init__(self, S = 7, C = 20, B = 2):
        self.S = S
        self.C = C
        self.B = B
        # Stage 1
        self.conv_layer = Conv_layer(filters=64, kernel_size=7, strides=2)
        self.BN_layer = BatchNorm()
        self.ReLU_layer = LeakyReLU()
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
        #  Final Stage
        self.avg_pool_layer = AVG_pooling_global()
        self.Flatten_layer = Flatten()
        self.FC_layer2 = Fully_connected(S * S * (C + B * 5))
        self.output = None

    def zero_pad(self, X, pad):
        X_pad = cp.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
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
        # Final Stage
        X = self.avg_pool_layer.forward(X)
        X = self.Flatten_layer.forward(X)
        X = self.FC_layer2.forward(X)
        X = X.reshape(-1,self.S, self.S, self.C + self.B * 5 )
        X[...,:20] = self.FC_layer2.softmax(X[...,:20])
        return X

    def backward(self, dZ_input, num_of_mn_batch, epoch):
        dZ = dZ_input.reshape(dZ_input.shape[0], -1)
        dZ = self.FC_layer2.backward(dZ, num_of_mn_batch, epoch)
        dZ = self.Flatten_layer.backward(dZ)
        dZ = self.avg_pool_layer.backward(dZ)

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






