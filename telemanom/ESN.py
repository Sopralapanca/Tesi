import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential
import os
import random


def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(x,y):
  feature = {
      'x': _bytes_feature(tf.io.serialize_tensor(x)),
      'y': _bytes_feature(tf.io.serialize_tensor(y)),
  }

  # Create a Features message using tf.train.Example.

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def read_tfrecord(example):
    tfrecord_format = (
        {
            "x": tf.io.FixedLenFeature([], tf.string),
            "y": tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)

    x = tf.io.parse_tensor(example['x'], out_type=tf.float32)
    y = tf.io.parse_tensor(example['y'], out_type=tf.double)

    return x, y

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

class ReservoirCell(keras.layers.Layer):
    # Implementation of a shallow reservoir to be used as cell of a Recurrent Neural Network
    # The implementation is parametrized by:
    # units - the number of recurrent neurons in the reservoir
    # input_scaling - the max abs value of a weight in the input-reservoir connections
    #                 note that whis value also scales the unitary input bias
    # spectral_radius - the max abs eigenvalue of the recurrent weight matrix
    # leaky - the leaking rate constant of the reservoir
    # connectivity_input - number of outgoing connections from each input unit to the reservoir

    def __init__(self, units, SEED,
                 input_scaling=1., spectral_radius=0.99, leaky=1,
                 connectivity_input=10,
                 **kwargs):

        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_input = connectivity_input
        self.SEED = SEED
        super().__init__(**kwargs)

    def build(self, input_shape):
        # build the input weight matrix
        self.kernel = sparse_tensor(input_shape[-1], self.units, self.connectivity_input) * self.input_scaling
        # build the recurrent weight matrix
        # uses circular law to determine the values of the recurrent weight matrix
        value = (self.spectral_radius / np.sqrt(self.units)) * (6 / np.sqrt(12))
        W = tf.random.uniform(shape=(self.units, self.units), minval=-value, maxval=value, seed=self.SEED)
        self.recurrent_kernel = W

        self.bias = tf.random.uniform(shape=(self.units,), minval=-1, maxval=1) * self.input_scaling

        self.built = True

    def call(self, inputs, states):
        # computes the output of the cell given the input and previous state
        prev_output = states[0]
        input_part = tf.sparse.sparse_dense_matmul(inputs, self.kernel)
        state_part = tf.matmul(prev_output, self.recurrent_kernel)
        output = prev_output * (1 - self.leaky) + tf.nn.tanh(input_part + self.bias + state_part) * self.leaky

        return output, [output]

    def get_config(self):
        base_config = super().get_config()

        return {**base_config,
                "units": self.units,
                "spectral_radius": self.spectral_radius,
                "leaky": self.leaky,
                "input_scaling": self.input_scaling,
                "connectivity_input": self.connectivity_input,
                "state_size": self.state_size
                }

    def from_config(cls, config):
        return cls(**config)

class SimpleESN(keras.Model):
    def __init__(self,units=100, input_scaling=1,
                 spectral_radius=0.99, leaky=1,connectivity_input=10,
                 config = None, SEED=42,
                 **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.connectivity_input = connectivity_input
        self.units = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky

        self.SEED = SEED

        if self.SEED == 42:
            os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        os.environ['PYTHONHASHSEED'] = str(self.SEED)

        random.seed(self.SEED)
        np.random.seed(self.SEED)
        tf.random.set_seed(self.SEED)


        self.reservoir = Sequential()
        self.reservoir.add(tf.keras.layers.RNN(cell=ReservoirCell(
                                                units=units,
                                                spectral_radius=spectral_radius, leaky=leaky,
                                                connectivity_input = connectivity_input,
                                                input_scaling=input_scaling,
                                                SEED=self.SEED)
                                              )
                           )

        self.readout = Sequential()
        self.readout.add(tf.keras.layers.Dense(config.n_predictions))
        self.readout.compile(loss=config.loss_metric, optimizer=config.optimizer)

    def call(self, inputs):
        r = self.reservoir(inputs)
        y = self.readout(r)
        return y

    def get_config(self):
        return {"connectivity_input": self.connectivity_input,
                "units": self.units,
                "input_scaling": self.input_scaling,
                "spectral_radius": self.spectral_radius,
                "leaky": self.leaky,
                "config": self.config}

    def from_config(cls, config):
        return cls(**config)

    def generator(self, dataset, batch_size):
        ds = dataset.repeat().prefetch(tf.data.AUTOTUNE)
        iterator = iter(ds)
        x, y = iterator.get_next()

        while True:
            yield x, y

    def fit(self, x, y, **kwargs):
        #PER MIGLIORARE I TEMPI LA SCRITTURA SU FILE SI POTREBBE FARE SOLO PER ALCUNI
        N = self.config.esn_batch_number
        training_steps = x.shape[0]//N
        train_reservoir = "./temp_files/train_reservoir.tfrecord"

        with tf.io.TFRecordWriter(train_reservoir) as file_writer:
            for i in range(N):
                X_train = self.reservoir(x[i * training_steps:(i + 1) * training_steps])
                y_train = y[i * training_steps:(i + 1) * training_steps]

                example = serialize_example(X_train, y_train)

                file_writer.write(example)

        #validation data
        x_val, y_val = kwargs['validation_data']
        validation_steps = x_val.shape[0] // N
        valid_reservoir = "./temp_files/valid_reservoir.tfrecord"

        #channel d-12 has validation shape (9,250,25)
        if validation_steps == 0:
            train_ds = (tf.data.TFRecordDataset(train_reservoir)
                  .map(read_tfrecord))
            iterator = train_ds.repeat().prefetch(tf.data.AUTOTUNE).as_numpy_iterator()

            x_val_1 = self.reservoir(x_val)
            kwargs['validation_data'] = (x_val_1, y_val)
            return self.readout.fit(iterator, steps_per_epoch=N, **kwargs)

        else:
            with tf.io.TFRecordWriter(valid_reservoir) as file_writer:
                for i in range(N):
                    x_val_1 = self.reservoir(x_val[i * validation_steps:(i + 1) * validation_steps])
                    y_val_1 = y_val[i * validation_steps:(i + 1) * validation_steps]

                    example = serialize_example(x_val_1, y_val_1)

                    file_writer.write(example)

        # reading tfrecord files
        train_dataset = tf.data.TFRecordDataset(train_reservoir).map(read_tfrecord)
        train_ds = self.generator(train_dataset, N)
        validation_dataset = tf.data.TFRecordDataset(valid_reservoir).map(read_tfrecord)
        valid_ds = self.generator(validation_dataset, N)

        kwargs['validation_data'] = (valid_ds)


        return self.readout.fit(train_ds, steps_per_epoch=N, validation_steps = validation_steps, **kwargs)