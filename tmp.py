import os
from telemanom.ESN import SimpleESN
import yaml
from datetime import datetime as dt
import pandas as pd
from telemanom.channel import Channel
from telemanom.helpers import Config
import time
from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner.tuners import Hyperband, RandomSearch
import csv


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

config_path = "config.yaml"
config = Config(config_path)

chan_df = pd.read_csv("labeled_anomalies.csv")

for i, row in chan_df.iterrows():
    chan_id = row.chan_id

    if chan_id != "P-10":
        continue

    channel = Channel(config, row.chan_id)
    channel.load_data()

    datset_size = len(channel.X_train)

    train_size = int((1 - config.validation_split) * datset_size)
    valid_size = int(config.validation_split * datset_size)

    print("dataset size", datset_size)
    print("train size", train_size)
    print("valid size", valid_size)

    print("calling model")

    model = SimpleESN(config= config,
                      inputs_shape=(None, channel.X_train.shape[2]),
                      units=200,
                      input_scaling=1,
                      spectral_radius=0.99,
                      leaky=0.8,
                      datset_size = datset_size,
                      train_size = train_size,
                      valid_size = valid_size
                      )

    print("calling compile")
    model.compile(loss=config.loss_metric, optimizer=config.optimizer, run_eagerly=True)

    print("calling fit")
    model.fit(channel.X_train,
              channel.y_train,
              batch_size=config.lstm_batch_size,
              epochs=30,
              validation_split=config.validation_split,
              verbose=1)



