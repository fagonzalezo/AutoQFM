import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import encoding
import qsvm


def metricas_modelos(y_true, y_pred):
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def Dataset(X, y, test_size_split=0.2):
    train_sample, test_sample, train_label, test_label = train_test_split(
        X, y, stratify=y, test_size=test_size_split, random_state=12
    )

    std_scale = StandardScaler().fit(train_sample)
    train_sample = std_scale.transform(train_sample)
    test_sample = std_scale.transform(test_sample)

    samples = np.append(train_sample, test_sample, axis=0)
    minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
    train_sample = minmax_scale.transform(train_sample)
    test_sample = minmax_scale.transform(test_sample)

    return train_sample, train_label, test_sample, test_label


class Fitness:
    def __init__(self, nqubits, nparameters, X, y, debug=False):
        self.nqubits = nqubits
        self.nparameters = nparameters
        self.cc = encoding.CircuitConversor(nqubits, nparameters)
        self.X = X
        self.y = y
        self.debug = debug

    def __call__(self, POP):
        return self.fitness(POP)

    def fitness(self, POP):
        # Convertimos el individuo en el fenotipo (ansatz)
        training_features, training_labels, test_features, test_labels = Dataset(
            self.X, self.y
        )
        model = qsvm.QSVM(
            lambda parameters: self.cc(POP, parameters)[0],
            training_features,
            training_labels,
        )
        y_pred = model.predict(
            test_features
        )  # 22% del computo (ver abajo line-profiler)
        acc = metricas_modelos(test_labels, y_pred)  # sklearn
        POP = "".join(str(i) for i in POP)
        _, gates = self.cc(POP, training_features[:, [0, 1]])
        if self.debug:
            print(f"String: {POP}\n -> accuracy = {acc}, gates = {gates}")
        gate = gates / self.nqubits
        wc = gate + (gate * (acc**2))
        return wc, acc  #


class CoherentFitness:
    def __init__(
        self,
        nqubits: int,
        nparameters: int,
        random_seed: int,
        sample_size: int,
        sigma: float = 1.0,
    ):
        self.nqubits = nqubits
        self.nparameters = nparameters
        self.cc = encoding.CircuitConversor(nqubits, nparameters)
        self.rng = np.random.default_rng(random_seed)
        self.sample_size = sample_size
        self.sigma = sigma

    def __call__(self, POP):
        uniform_x = self.rng.uniform(-1, 1, size=(self.nparameters, self.sample_size))
        uniform_y = self.rng.uniform(-1, 1, size=(self.nparameters, self.sample_size))
        target_kernel_values = self.gaussian_kernel(uniform_x, uniform_y, self.sigma)

        x_states, x_n_gates = self.cc(POP, uniform_x)
        y_states, y_n_gates = self.cc(POP, uniform_y)
        assert x_n_gates == y_n_gates

        normalising_constant = 1  # (2 * np.pi * self.sigma) ** (-self.nparameters / 2)
        predicted_kernel_values = normalising_constant * np.array(
            [np.abs(np.dot(x, np.conj(y))) ** 2 for (x, y) in zip(x_states, y_states)]
        )

        mse = np.mean((target_kernel_values - predicted_kernel_values) ** 2)
        avg_gates = (x_n_gates + y_n_gates) / 2
        gate = avg_gates / self.nqubits
        return gate, mse

    @staticmethod
    def gaussian_kernel(x, y, sigma):
        # First dimension is feature dimension, second dimension is sample dimension
        return np.exp(-np.linalg.norm(x - y, axis=0) ** 2 / (2 * (sigma**2)))
