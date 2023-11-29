import os
import numpy as np
from sklearn.mixture import GaussianMixture
import torch.nn as nn


class NumericalField:
    data_type = "Numerical Data"

    def __init__(self, model="gmm", n=5):
        self.sigma = None
        self.mu = None
        self.K = None
        self.Max = None
        self.Min = None
        self.stds = None
        self.means = None
        self.weights = None
        self.model_name = model
        if model == "gmm":
            self.n = n
            self.model = GaussianMixture(n)
        elif model == "minmax":
            pass
        self.fit_data = None
        self.learned = False

    def learn(self):
        if self.learned:
            return
        if self.model_name == "gmm":
            self.model.fit(self.fit_data)
            self.weights = self.model.weights_
            self.means = self.model.means_.reshape((1, self.n))[0]
            self.stds = np.sqrt(self.model.covariances_).reshape((1, self.n))[0]
        elif self.model_name == "minmax":
            self.Min = np.min(self.fit_data)
            self.Max = np.max(self.fit_data)
            self.K = 1 / (self.Max - self.Min)
        elif self.model_name == 'mean_std':
            self.mu = np.mean(self.fit_data)
            self.sigma = np.std(self.fit_data)
        else:
            print('invalid encoding method')
        self.learned = True

    def get_data(self, data):
        if self.fit_data is None:
            self.fit_data = data
        else:
            self.fit_data = np.concatenate([self.fit_data, data], axis=0)

    # 处理数据
    def convert(self, data):
        assert isinstance(data, np.ndarray)
        features = None
        if self.model_name == "gmm":
            features = (data - self.means) / (2 * self.stds)
            probs = self.model.predict_proba(data)
            argmax = np.argmax(probs, axis=1)
            idx = np.arange(len(features))
            features = features[idx, argmax].reshape(-1, 1)
            features = np.concatenate((features, probs), axis=1)
        elif self.model_name == "minmax":
            features = self.K * (data - self.Min)
        elif self.model_name == 'mean_std':
            features = (data - self.mu) / self.sigma
        else:
            print('invalid encoding method')
        return features

    # 数据还原
    def reverse(self, features):
        assert isinstance(features, np.ndarray)
        data = None
        if self.model_name == "gmm":
            assert features.shape[1] == self.n + 1
            v = features[:, 0]
            u = features[:, 1:self.n + 1].reshape(-1, self.n)
            argmax = np.argmax(u, axis=1)
            mean = self.means[argmax]
            std = self.stds[argmax]
            v_ = v * 2 * std + mean
            data = v_.reshape(-1, 1)
        elif self.model_name == "minmax":
            data = features / self.K + self.Min
        elif self.model_name == "mean_std":
            data = features * self.sigma + self.mu
        else:
            print('invalid encoding method')
        return data

    def dim(self):
        if self.model_name == "gmm":
            return self.n + 1
        else:
            return 1


class CategoricalField:
    data_type = "Categorical Data"

    def __init__(self, method="one-hot", noise=0.2):
        self.dict = {}
        self.rev_dict = {}
        self.method = method
        self.fit_data = None
        self.noise = noise

    def learn(self):
        vst = np.unique(self.fit_data)
        for idx, v in enumerate(vst):
            self.dict[v] = idx
            self.rev_dict[idx] = v

    def get_data(self, data):
        if self.fit_data is None:
            self.fit_data = data
        else:
            self.fit_data = np.concatenate([self.fit_data, data], axis=0)

    def convert(self, data):
        assert isinstance(data, np.ndarray)
        data = data.reshape(1, -1)
        data = list(map(lambda x: self.dict[x], data[0]))
        data = np.asarray(data).reshape(-1, 1)
        features = data
        if self.method == "dict":
            return features

        if self.method == "one-hot":
            features = np.zeros((data.shape[0], len(self.dict)), dtype="int")
            idx = np.arange(len(features))
            features[idx, data.reshape(1, -1)] = 1
            if self.noise is not None:
                noise = np.random.uniform(0, self.noise, features.shape)
                features = features + noise
                temp = np.sum(features, axis=1).reshape(-1, 1)
                features = features / temp

        if self.method == "embedding":
            embedding_layer = nn.Embedding(len(self.dict), len(self.dict))
            features = embedding_layer(data)

        return features

    # 将编码的数据转回具体的数值
    def reverse(self, features):
        assert isinstance(features, np.ndarray)
        if self.method == "one-hot":
            assert features.shape[1] == len(self.dict)
            features = np.argmax(features, axis=1)

        row_num = features.shape[0]
        features = features.reshape(1, -1)
        data = list(map(lambda x: self.rev_dict[x], features[0]))
        data = np.asarray(data)
        data = data.reshape((row_num, 1))
        return data

    def dim(self):
        return 1 if self.method == "dict" else len(self.dict)


def creatFile(name):
    try:
        os.mkdir("../exp-dir")
    except FileExistsError:
        pass
    path = "exp-dir/" + name + "/"
    try:
        os.mkdir("exp-dir/" + name)
    except FileExistsError:
        pass
