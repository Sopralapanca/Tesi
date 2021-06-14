from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from telemanom.DeepESN import SimpleDeepReservoirLayer
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
    model = Sequential()
    if len(hp) == 0:
        model.add(SimpleDeepReservoirLayer(input_shape=(None, channel.X_train.shape[2])))
    else:
        model.add(SimpleDeepReservoirLayer(input_shape=(None, channel.X_train.shape[2]),
                                           units=int(hp["units"]),
                                           return_sequences=(hp["return_sequences"] == 'true'),
                                           layers=int(hp["layers"]),
                                           concat=hp["concat"] == 'true',
                                           input_scaling=float(hp["input_scaling"]),
                                           inter_scaling=float(hp["inter_scaling"]),
                                           spectral_radius=float(hp["radius"]),
                                           leaky=float(hp["leaky"]),
                                           connectivity_recurrent=int(hp["connectivity_recurrent"]),
                                           connectivity_inter=int(hp["connectivity_inter"]),
                                           connectivity_input=int(hp["connectivity_input"])
                                           )
                  )

    model.add(Dense(config.n_predictions))
    model.add(Activation('linear'))

    return model