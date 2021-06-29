from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from telemanom.DeepESN import SimpleDeepReservoirLayer
from telemanom.ESN import SimpleESN
import tensorflow as tf
import os

def create_lstm_model(channel,config):
    model = Sequential()

    model.add(LSTM(
        config.layers[0],
        input_shape=(None, channel.X_train.shape[2]),
        return_sequences=True))
    model.add(Dropout(config.dropout))

    model.add(LSTM(
        config.layers[1],
        return_sequences=False))
    model.add(Dropout(config.dropout))

    model.add(Dense(
        config.n_predictions))
    model.add(Activation('linear'))

    return model

def create_esn_model(channel,config, hp):
    if len(hp) == 0:
        model = SimpleESN(inputs_shape=(None, channel.X_train.shape[2]),
                          config=config
                          )
    else:
        model = SimpleESN(inputs_shape=(None, channel.X_train.shape[2]),
                          config=config,
                          units=int(hp["units"]),
                          return_sequences=(hp["return_sequences"] == 'true'),
                          input_scaling=float(hp["input_scaling"]),
                          spectral_radius=float(hp["radius"]),
                          leaky=float(hp["leaky"]),
                          connectivity_recurrent=int(hp["connectivity_recurrent"]),
                          connectivity_input=int(hp["connectivity_input"])
                          )

        model.compile(loss=config.loss_metric,
                      optimizer=config.optimizer)

        return model