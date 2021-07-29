from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from telemanom.ESN import SimpleESN
import random

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

    model.compile(loss=config.loss_metric,
                       optimizer=config.optimizer)
    return model

def create_esn_model(channel,config, hp, seed):
    if len(hp) == 0:
        model = SimpleESN(config=config,
                          SEED=seed
                          )
    else:
        model = SimpleESN(config=config,
                          units=int(hp["units"]),
                          input_scaling=float(hp["input_scaling"]),
                          spectral_radius=float(hp["radius"]),
                          leaky=float(hp["leaky"]),
                          connectivity_input=int(hp["connectivity_input"]),
                          SEED=seed
                          )

    model.build(input_shape=(channel.X_train.shape[0],channel.X_train.shape[1],channel.X_train.shape[2]))
    model.compile(loss=config.loss_metric,
                  optimizer=config.optimizer)

    return model