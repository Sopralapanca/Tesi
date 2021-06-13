import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from telemanom.DeepESN import SimpleDeepReservoirLayer
import tensorflow as tf
import yaml
from datetime import datetime as dt
import logging
import pandas as pd
from numpy import arange
from telemanom.channel import Channel
from telemanom.helpers import Config
import numpy as np
import decimal

SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

config_path = "config.yaml"
config = Config(config_path)

data = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
os.mkdir('hp/{}'.format(data))
os.mkdir('hp/{}/config/'.format(data))
os.mkdir('hp/{}/weights/'.format(data))

logger = logging.getLogger('find_hp')
logger.setLevel(logging.INFO)

stdout = logging.StreamHandler(sys.stdout)
stdout.setLevel(logging.INFO)
logger.addHandler(stdout)
# add logging FileHandler based on ID
hdlr = logging.FileHandler('hp/%s/params_list.log' % data)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)

try:
    infile = os.path.join("logs", "{}.log".format(config.logs_id))
except (FileNotFoundError, OSError) as e:
    raise e

important = []
lossess = []
keep_phrases = ["validation_loss"]

##### da rivedere ####
"""with open(infile) as f:
    f = f.readlines()

for line in f:
    for phrase in keep_phrases:
        if phrase in line:
            important.append(line)
            break

for elem in important:
    tmp = elem.split("INFO")[1]
    string = tmp.split(":")[1].replace(" ", "").strip()
    lossess.append(float(string))"""


###############################
#settare per quando non ci sono label
chan_df = pd.read_csv("labeled_anomalies.csv")

logger.info("List of tested parameters")
def float_range(start, stop, step):
  while start < stop:
    yield float(start)
    start += float(decimal.Decimal(step))

for i, row in chan_df.iterrows():
    if config.tune_hp:

        #carica i valori
        path = os.path.join("hp", config.hp_id, "{}.yaml".format(row.chan_id))
        with open(path, 'r') as file:
            hp = yaml.load(file, Loader=yaml.BaseLoader)

        radius = float(hp["radius"])
        l = float(hp["leaky"])
        in_scaling = float(hp["input_scaling"])
        int_scaling = float(hp["inter_scaling"])
        n_units = int(hp["units"])
        n_layers = int(hp["layers"])
        concat = [True, False]

        #genera nuovi valori
        units = [i for i in range((n_units-50), (n_units+50), 25)]
        if n_layers > 1:
            layers = [(n_layers-1),(n_layers), (n_layers+1)]
        else:
            layers = [(n_layers),(n_layers + 1)]

        #controllare se posso usare input scaling maggiore di 1
        if in_scaling < 1:
            input_scaling = list(float_range((in_scaling - 0.25), (in_scaling + 0.25), 0.10))
        else:
            input_scaling = [in_scaling]

        if int_scaling < 1:
            inter_scaling = list(float_range((int_scaling - 0.25), (int_scaling + 0.25), 0.10))
        else:
            inter_scaling = [int_scaling]

        if radius == 0.99:
            spectral_radius = list(float_range((radius - 0.40), radius, 0.10))
        else:
            spectral_radius = list(float_range((radius - 0.25), (radius + 0.25), 0.10))

        if l == 1.0:
            leaky = list(float_range((l - 0.40), l, 0.10))
        else:
            leaky = list(float_range((l - 0.25), (l + 0.25), 0.10))

    else:
        spectral_radius = [0.40, 0.75, 0.99]
        leaky = [0.40, 0.75, 1.]
        input_scaling = [0.25, 0.50, 0.75, 1.]
        inter_scaling = [0.25, 0.50, 0.75, 1.]
        units = [50, 350, 500, 700]
        layers = [1, 3, 5]
        concat = [True, False]

    logger.info('Stream # {}: {}'.format(i+1, row.chan_id))
    logger.info("spectral_radius: {}".format(spectral_radius))
    logger.info("leaky: {}".format(leaky))
    logger.info("input_scaling: {}".format(input_scaling))
    logger.info("inter_scaling: {}".format(inter_scaling))
    logger.info("units: {}".format(units))
    logger.info("layers: {}\n".format(layers))


    # default hp
    hp = {
        'units': 50,
        'layers': 1,
        'concat': False,
        'input_scaling': 1,
        'inter_scaling': 1,
        'radius': 0.99,
        'leaky': 1,
        'connectivity_recurrent': 10,
        'connectivity_input': 10,
        'connectivity_inter': 10,
        'return_sequences': False
    }

    min_val_loss = 10000
    min_rad = 10000
    min_leaky = 10000
    min_in_scaling = 10000
    min_inter_scaling = 10000

    hidden_layers = 0
    reservoir_units = 0
    concatenation = False

    channel = Channel(config, row.chan_id)
    channel.load_data()

    print("searching for input scaling")
    for elem in input_scaling:
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(SimpleDeepReservoirLayer(input_shape=(None, channel.X_train.shape[2]),
                                           units=hp["units"],
                                           return_sequences=hp["return_sequences"],
                                           layers=hp["layers"],
                                           concat=hp["concat"],
                                           input_scaling=elem,
                                           inter_scaling=hp["inter_scaling"],
                                           spectral_radius=hp["radius"],
                                           leaky=hp["leaky"],
                                           connectivity_recurrent=hp["connectivity_recurrent"],
                                           connectivity_inter=hp["connectivity_inter"],
                                           connectivity_input=hp["connectivity_input"]
                                           )
                  )

        model.add(Dense(config.n_predictions))
        model.add(Activation('linear'))

        model.compile(loss=config.loss_metric,
                      optimizer=config.optimizer)

        history = model.fit(channel.X_train,
                            channel.y_train,
                            batch_size=config.lstm_batch_size,
                            epochs=5,
                            validation_split=config.validation_split,
                            verbose=1)

        val_loss = history.history["val_loss"][-1]

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_in_scaling = elem

    hp["input_scaling"] = min_in_scaling
    min_val_loss = 10000

    print("searching for spectral radius")
    for rad in spectral_radius:
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(SimpleDeepReservoirLayer(input_shape=(None, channel.X_train.shape[2]),
                                           units=hp["units"],
                                           return_sequences=hp["return_sequences"],
                                           layers=hp["layers"],
                                           concat=hp["concat"],
                                           input_scaling=hp["input_scaling"],
                                           inter_scaling=hp["inter_scaling"],
                                           spectral_radius=rad,
                                           leaky=hp["leaky"],
                                           connectivity_recurrent=hp["connectivity_recurrent"],
                                           connectivity_inter=hp["connectivity_inter"],
                                           connectivity_input=hp["connectivity_input"]
                                           )
                  )

        model.add(Dense(config.n_predictions))
        model.add(Activation('linear'))

        model.compile(loss=config.loss_metric,
                      optimizer=config.optimizer)

        history = model.fit(channel.X_train,
                            channel.y_train,
                            batch_size=config.lstm_batch_size,
                            epochs=5,
                            validation_split=config.validation_split,
                            verbose=1)

        val_loss = history.history["val_loss"][-1]

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_rad = rad

    hp["radius"] = min_rad
    min_val_loss = 10000

    print("searching for leaky")
    for elem in leaky:
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(SimpleDeepReservoirLayer(input_shape=(None, channel.X_train.shape[2]),
                                           units=hp["units"],
                                           return_sequences=hp["return_sequences"],
                                           layers=hp["layers"],
                                           concat=hp["concat"],
                                           input_scaling=hp["input_scaling"],
                                           inter_scaling=hp["inter_scaling"],
                                           spectral_radius=hp["radius"],
                                           leaky=elem,
                                           connectivity_recurrent=hp["connectivity_recurrent"],
                                           connectivity_inter=hp["connectivity_inter"],
                                           connectivity_input=hp["connectivity_input"]
                                           )
                  )

        model.add(Dense(config.n_predictions))
        model.add(Activation('linear'))

        model.compile(loss=config.loss_metric,
                      optimizer=config.optimizer)

        history = model.fit(channel.X_train,
                            channel.y_train,
                            batch_size=config.lstm_batch_size,
                            epochs=5,
                            validation_split=config.validation_split,
                            verbose=1)

        val_loss = history.history["val_loss"][-1]

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_leaky = elem

    hp["leaky"] = min_leaky
    min_val_loss = 10000

    print("searching for layers and units")
    save_model = None
    for n_layers in layers:
        for elem in units:
            for el in concat:
                if n_layers == 1 and el is True:
                    continue

                model = Sequential()

                model.add(SimpleDeepReservoirLayer(input_shape=(None, channel.X_train.shape[2]),
                                                   units=elem,
                                                   return_sequences=hp["return_sequences"],
                                                   layers=n_layers,
                                                   concat=el,
                                                   input_scaling=hp["input_scaling"],
                                                   inter_scaling=hp["inter_scaling"],
                                                   spectral_radius=hp["radius"],
                                                   leaky=hp["leaky"],
                                                   connectivity_recurrent=hp["connectivity_recurrent"],
                                                   connectivity_inter=hp["connectivity_inter"],
                                                   connectivity_input=hp["connectivity_input"]
                                                   )
                          )

                model.add(Dense(config.n_predictions))
                model.add(Activation('linear'))

                model.compile(loss=config.loss_metric,
                              optimizer=config.optimizer)

                history = model.fit(channel.X_train,
                                    channel.y_train,
                                    batch_size=config.lstm_batch_size,
                                    epochs=5,
                                    validation_split=config.validation_split,
                                    verbose=1)

                val_loss = history.history["val_loss"][-1]

                if val_loss <= min_val_loss:
                    min_val_loss = val_loss
                    reservoir_units = elem
                    hidden_layers = n_layers
                    concatenation = el
                    save_model = model

    hp["units"] = reservoir_units
    hp["layers"] = hidden_layers
    hp["concat"] = concatenation

    min_val_loss = 10000
    if hp["layers"] > 1:
        print("searching for inter scaling")
        for elem in inter_scaling:
            tf.keras.backend.clear_session()
            model = Sequential()
            model.add(SimpleDeepReservoirLayer(input_shape=(None, channel.X_train.shape[2]),
                                               units=hp["units"],
                                               return_sequences=hp["return_sequences"],
                                               layers=hp["layers"],
                                               concat=hp["concat"],
                                               input_scaling=hp["input_scaling"],
                                               inter_scaling=elem,
                                               spectral_radius=hp["radius"],
                                               leaky=hp["leaky"],
                                               connectivity_recurrent=hp["connectivity_recurrent"],
                                               connectivity_inter=hp["connectivity_inter"],
                                               connectivity_input=hp["connectivity_input"]
                                               )
                      )

            model.add(Dense(config.n_predictions))
            model.add(Activation('linear'))

            model.compile(loss=config.loss_metric,
                          optimizer=config.optimizer)

            history = model.fit(channel.X_train,
                                channel.y_train,
                                batch_size=config.lstm_batch_size,
                                epochs=5,
                                validation_split=config.validation_split,
                                verbose=1)

            val_loss = history.history["val_loss"][-1]

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                min_inter_scaling = elem
                save_model = model

        hp["inter_scaling"] = min_inter_scaling

    hp["validation_loss"] = val_loss

    f = open(os.path.join('hp/config',data, '{}.yaml'.format(row.chan_id)), "w")

    yaml.dump(hp, f, default_flow_style=False)
    f.close()

    save_model.save_weights(os.path.join("/hp/weights", '{}_weights.h5'.format(channel)))
