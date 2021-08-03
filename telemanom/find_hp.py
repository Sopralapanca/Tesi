import os
from telemanom.ESN import SimpleESN
import yaml
import time
from keras_tuner.engine.hypermodel import HyperModel
from keras_tuner.tuners import Hyperband, RandomSearch
import csv

import tensorflow as tf


class MyHyperModel(HyperModel):
    def __init__(self, config, channel):
        self.config = config
        self.channel = channel


    def build(self, hp):
        model = SimpleESN(config=self.config,
                          units=hp.Int("units", 100, 1000, 100),
                          input_scaling=hp.Float("input_scaling", 0.5, 1, 0.10),
                          spectral_radius=hp.Choice("spectral_radius",
                                                    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]),
                          leaky=hp.Float("leaky", 0.1, 1, 0.10),
                          SEED=42
                          )

        model.build(input_shape=(self.channel.X_train.shape[0], self.channel.X_train.shape[1], self.channel.X_train.shape[2]))
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(loss=self.config.loss_metric,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate))

        return model

class FindHP():
    def __init__(self, id, channel, config):
        self.id = id
        self.channel = channel
        self.config = config

        self.run()

    def run(self):
        # default hp
        hp = {
            'units': 100,
            'input_scaling': 1,
            'radius': 0.99,
            'leaky': 1,
            'connectivity_input': 10
        }

        tuner = RandomSearch(
            MyHyperModel(config=self.config, channel = self.channel),
            objective="val_loss",
            directory='hp/{}/kerastuner'.format(self.id),
            max_trials=30,
            seed=42,
            project_name=self.channel.id,
        )


        tuner.search(self.channel.X_train,
                     self.channel.y_train,
                     batch_size=self.config.esn_batch_number,
                     epochs=10,
                     validation_data=(self.channel.X_valid, self.channel.y_valid),
                     verbose=1
                     )


        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        hp["units"] = best_hps.get('units')
        hp["input_scaling"] = float("{:.2f}".format(best_hps.get('input_scaling')))
        hp["radius"] = float("{:.2f}".format(best_hps.get('spectral_radius')))
        hp["leaky"] = float("{:.2f}".format(best_hps.get('leaky')))
        hp["learning_rate"] = float(format(best_hps.get('learning_rate')))
        f = open(f'./hp/{self.id}/config/{self.channel.id}.yaml', "w")

        yaml.dump(hp, f, default_flow_style=False)
        f.close()