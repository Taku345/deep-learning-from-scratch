import sys, os

sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        """_summary_

        Args:
            input_size (int): 入力層の要素数
            hidden_size (_type_): _description_
            output_size (_type_): _description_
            weight_init_std (float, optional): 重みの初期値の標準偏差
        """
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

            return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # tがone hotの場合の対応
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"], grads["b1"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W2"], grads["b2"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads
