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

if config.resume_hp_search:
    data = config.hp_search_id
else:
    data = dt.now().strftime('%Y-%m-%d_%H.%M.%S')

try:
    os.mkdir('hp/{}'.format(data))
    os.mkdir('hp/{}/config/'.format(data))
    os.mkdir('hp/{}/kerastuner/'.format(data))
except FileExistsError as e:
    pass
except Exception as e:
    raise e


class MyHyperModel(HyperModel):
    def __init__(self, X_train, config):
        self.config = config
        self.X_train = X_train

    def build(self, hp):
        model = SimpleESN(inputs_shape=(None, self.X_train.shape[2]),
                          config=self.config,
                          units=hp.Int("units", 50, 350, 50),
                          input_scaling=hp.Float("input_scaling", 0.1, 1, 0.15),
                          spectral_radius=hp.Float("spectral_radius", 0.1, 1.10, 0.10),
                          leaky=hp.Float("leaky", 0.1, 1, 0.15),
                          )

        model.compile(loss=config.loss_metric,
                      optimizer=config.optimizer)
        return model


# Columns headers for output file
col_header = ["model", "total_elapsed_time"]
times_row = {
    'model': "ESN",
    'total_elapsed_time': 0
}

total_elapsed_time = 0

chan_df = pd.read_csv("labeled_anomalies.csv")

for i, row in chan_df.iterrows():
    print('Stream # {}: {}'.format(i + 1, row.chan_id))
    chan_id = row.chan_id

    # default hp
    hp = {
        'units': 50,
        'input_scaling': 1,
        'radius': 0.99,
        'leaky': 1,
        'connectivity_recurrent': 10,
        'connectivity_input': 10,
        'return_sequences': False
    }

    channel = Channel(config, row.chan_id)
    channel.load_data()

    dir = 'hp/{}/kerastuner/{}'.format(data, chan_id)
    # resume hyperparameter search
    if os.path.isdir(dir):
        file1 = open('hp/{}/times.log'.format(data), 'r')

        while True:
            line = file1.readline()

            # if line is empty end of file is reached
            if not line:
                break

            strings = line.strip().split(" ")
            channel_id = strings[0]

            if not channel_id == chan_id:
                continue

            if channel_id == chan_id:
                end_time = float(strings[1])
                break

        file1.close()

    else:

        tuner = RandomSearch(
            MyHyperModel(X_train=channel.X_train, config=config),
            objective="val_loss",
            directory='hp/{}/kerastuner'.format(data),
            max_trials=50,
            seed=42,
            project_name=row.chan_id,
        )

        start_time = time.time()
        tuner.search(channel.X_train,
                     channel.y_train,
                     batch_size=config.lstm_batch_size,
                     epochs=5,
                     validation_split=config.validation_split,
                     verbose=1
                     )
        end_time = (time.time() - start_time)

        f = open('hp/{}/times.log'.format(data), "a")
        f.write("{} {}\n".format(row.chan_id, end_time))
        f.close()

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        hp["units"] = best_hps.get('units')
        hp["input_scaling"] = float(best_hps.get('input_scaling'))
        hp["radius"] = float(best_hps.get('spectral_radius'))
        hp["leaky"] = float(best_hps.get('leaky'))

        f = open(os.path.join('hp', data, 'config/{}.yaml'.format(row.chan_id)), "w")

        yaml.dump(hp, f, default_flow_style=False)
        f.close()

    col_header.append(chan_id)
    times_row[chan_id] = end_time

    total_elapsed_time += end_time

    start_time = 0
    end_time = 0

times_row["total_elapsed_time"] = total_elapsed_time

with open('/hp/{}/times_ms.csv'.format(data), 'a') as filedata:
    writer = csv.DictWriter(filedata, delimiter=',', fieldnames=col_header)
    writer.writeheader()
    writer.writerow(times_row)

from functools import reduce


def secondsToStr(t):
    return "%dh:%02dm:%02ds.%03d" % \
           reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                  [(t * 1000,), 1000, 60, 60])


for key in times_row:
    if key == "model":
        continue
    else:
        times_row[key] = secondsToStr(float(times_row[key]))

with open('/hp/{}/times.csv'.format(data), 'a') as filedata:
    writer = csv.DictWriter(filedata, delimiter=',', fieldnames=col_header)
    writer.writeheader()
    writer.writerow(times_row)
