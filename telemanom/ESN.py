import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
import os
import random

SEED = 42

os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def sparse_eye(M):
    # Generates an M x M matrix to be used as sparse identity matrix for the
    # re-scaling of the sparse recurrent kernel in presence of non-zero leakage.
    # The neurons are connected according to a ring topology, where each neuron
    # receives input only from one neuron and propagates its activation only to one other neuron.
    # All the non-zero elements are set to 1
    dense_shape = (M, M)

    # gives the shape of a ring matrix:
    indices = np.zeros((M, 2))
    for i in range(M):
        indices[i, :] = [i, i]
    values = np.ones(shape=(M,)).astype('f')

    W = (tf.sparse.reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)))
    return W


def sparse_tensor(M, N, C=1):
    # Generates an M x N matrix to be used as sparse (input) kernel
    # For each row only C elements are non-zero
    # (i.e., each input dimension is projected only to C neurons).
    # The non-zero elements are generated randomly from a uniform distribution in [-1,1]

    dense_shape = (M, N)  # the shape of the dense version of the matrix

    indices = np.zeros((M * C, 2))  # indices of non-zero elements initialization
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th row of the matrix
        idx = np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k, :] = [i, idx[j]]
            k = k + 1
    values = 2 * (2 * np.random.rand(M * C).astype('f') - 1)
    W = (tf.sparse.reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)))
    return W


def sparse_recurrent_tensor(M, C=1):
    # Generates an M x M matrix to be used as sparse recurrent kernel
    # For each column only C elements are non-zero
    # (i.e., each recurrent neuron takes input from C other recurrent neurons).
    # The non-zero elements are generated randomly from a uniform distribution in [-1,1]

    dense_shape = (M, M)  # the shape of the dense version of the matrix

    indices = np.zeros((M * C, 2))  # indices of non-zero elements initialization
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th column of the matrix
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, :] = [idx[j], i]
            k = k + 1
    values = 2 * (2 * np.random.rand(M * C).astype('f') - 1)
    W = (tf.sparse.reorder(tf.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)))
    return W


class ReservoirCell(keras.layers.Layer):
    # Implementation of a shallow reservoir to be used as cell of a Recurrent Neural Network
    # The implementation is parametrized by:
    # units - the number of recurrent neurons in the reservoir
    # input_scaling - the max abs value of a weight in the input-reservoir connections
    #                 note that whis value also scales the unitary input bias
    # spectral_radius - the max abs eigenvalue of the recurrent weight matrix
    # leaky - the leaking rate constant of the reservoir
    # connectivity_input - number of outgoing connections from each input unit to the reservoir
    # connectivity_recurrent - number of incoming recurrent connections for each reservoir unit

    def __init__(self, units,
                 input_scaling=1., spectral_radius=0.99, leaky=1,
                 connectivity_input=10, connectivity_recurrent=10,
                 **kwargs):

        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_input = connectivity_input
        self.connectivity_recurrent = connectivity_recurrent
        super().__init__(**kwargs)

    def build(self, input_shape):

        # build the input weight matrix
        self.kernel = sparse_tensor(input_shape[-1], self.units, self.connectivity_input) * self.input_scaling

        # build the recurrent weight matrix
        W = sparse_recurrent_tensor(self.units, C=self.connectivity_recurrent)

        # re-scale the weight matrix to control the effective spectral radius of the linearized system
        if (self.leaky == 1):
            # if no leakage then rescale the W matrix
            # compute the spectral radius of the randomly initialized matrix
            e, _ = tf.linalg.eig(tf.sparse.to_dense(W))
            rho = max(abs(e))
            # rescale the matrix to the desired spectral radius
            W = W * (self.spectral_radius / rho)
            self.recurrent_kernel = W
        else:
            I = sparse_eye(self.units)
            W2 = tf.sparse.add(I * (1 - self.leaky), W * self.leaky)
            e, _ = tf.linalg.eig(tf.sparse.to_dense(W2))
            rho = max(abs(e))
            W2 = W2 * (self.spectral_radius / rho)
            self.recurrent_kernel = tf.sparse.add(W2, I * (self.leaky - 1)) * (1 / self.leaky)

        self.bias = tf.random.uniform(shape=(self.units,), minval=-1, maxval=1) * self.input_scaling

        self.built = True

    def call(self, inputs, states):
        # computes the output of the cell given the input and previous state
        prev_output = states[0]

        input_part = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        state_part = tf.sparse.sparse_dense_matmul(prev_output, self.recurrent_kernel)
        output = prev_output * (1 - self.leaky) + tf.nn.tanh(input_part + self.bias + state_part) * self.leaky

        return output, [output]


class SimpleReservoirLayer(keras.layers.Layer):
    # A layer structure implementing the functionalities of a Reservoir.
    # The layer is parametrized by the following:
    # units - the number of recurrent units used in the neural network;
    # input_scaling - the scaling coefficient of the first reserovir level
    # spectral_radius - the spectral radius of all the reservoir levels
    # leaky - the leakage coefficient of all the reservoir levels
    # connectivity_input - input connectivity coefficient of the input weight matrix
    # connectivity_recurrent - recurrent connectivity coefficient of all the recurrent weight matrices
    # return_sequences - if True, the state is returned for each time step, otherwise only for the last time step

    def __init__(self, units=100,
                 input_scaling=1,
                 spectral_radius=0.99, leaky=1,
                 connectivity_recurrent=10,
                 connectivity_input=10,
                 return_sequences=False,
                 **kwargs):

        super().__init__(**kwargs)
        self.units = units

        self.reservoir = keras.layers.RNN(ReservoirCell(units=units,
                                                        input_scaling=input_scaling,
                                                        spectral_radius=spectral_radius,
                                                        leaky=leaky,
                                                        connectivity_input=connectivity_input,
                                                        connectivity_recurrent=connectivity_recurrent),
                                          return_sequences=True, return_state=True
                                          )

        self.return_sequences = return_sequences

    def call(self, inputs):
        # compute the output of the reservoir

        X = inputs  # external input
        layer_states, layer_states_last = self.reservoir(X)

        if self.return_sequences:
            # all the time steps for the last layer
            return layer_states
        else:
            # the last time step for the last layer
            return layer_states_last


class SimpleESN(keras.Model):
    def __init__(self, inputs_shape, config,
                 units=100,
                 input_scaling=1,
                 spectral_radius=0.99, leaky=1,
                 connectivity_recurrent=1,
                 connectivity_input=10,
                 return_sequences=False,

                 **kwargs):
        super().__init__(**kwargs)

        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        self.inputs_shape = inputs_shape
        self.config = config

        """self.reservoir = SimpleReservoirLayer(input_shape=(self.inputs_shape),
                                                  units=units,
                                                  spectral_radius=spectral_radius, leaky=leaky,
                                                  input_scaling=input_scaling,
                                                  connectivity_recurrent=connectivity_recurrent,
                                                  connectivity_input=connectivity_input,
                                                  return_sequences=return_sequences)"""

        self.reservoir = Sequential()
        self.reservoir.add(
            SimpleReservoirLayer(input_shape=(self.inputs_shape),
                                                  units=units,
                                                  spectral_radius=spectral_radius, leaky=leaky,
                                                  input_scaling=input_scaling,
                                                  connectivity_recurrent=connectivity_recurrent,
                                                  connectivity_input=connectivity_input,
                                                  return_sequences=return_sequences)

        )
        self.reservoir.compile(loss=self.config.loss_metric, optimizer=self.config.optimizer)

        #self.masking = tf.keras.layers.Masking()


        self.readout = Sequential()
        self.readout.add(tf.keras.layers.Dense(self.config.n_predictions))
        self.readout.compile(loss=self.config.loss_metric, optimizer=self.config.optimizer)

    def call(self, inputs):
        #m = self.masking(inputs)
        r = self.reservoir(inputs)
        y = self.readout(r)
        return y

    def fit(self, x, y, **kwargs):
        x_train_1 = self.reservoir(x)
        return self.readout.fit(x_train_1, y, **kwargs)
