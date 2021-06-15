import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from telemanom.DeepESN import SimpleDeepReservoirLayer
import tensorflow as tf
import yaml
from datetime import datetime as dt
import pandas as pd
from telemanom.channel import Channel
from telemanom.helpers import Config
import numpy as np

SEED = 42
os.environ['PYTHONHASHSEED']=str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

config_path = "config.yaml"
config = Config(config_path)

data = dt.now().strftime('%Y-%m-%d_%H.%M.%S')
os.mkdir('hp/{}'.format(data))
os.mkdir('hp/{}/config/'.format(data))
os.mkdir('hp/{}/weights/'.format(data))


def create_float_array(min, max, elems):
  array = np.linspace(min,max,elems).tolist()
  formatted_array  = [ round(elem, 2) for elem in array ]
  return formatted_array

def modeling(model):
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
    return history

##### da rivedere ####
"""
important = []
lossess = []
keep_phrases = ["validation_loss"]
with open(infile) as f:
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
#solo per labeled_anomalies, creare script anche per dataset senza label
chan_df = pd.read_csv("labeled_anomalies.csv")

for i, row in chan_df.iterrows():
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

    input_scaling = create_float_array(0.1, 1, 20)
    spectral_radius = create_float_array(0.1, 1, 20)
    leaky = create_float_array(0.1, 1, 20)
    units = list(range(50, 1000, 100))

    """for elem in input_scaling:
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
        
        history = modeling(model)
        
        val_loss = history.history["val_loss"][-1]

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_in_scaling = elem

    hp["input_scaling"] = min_in_scaling
    min_val_loss = 10000"""

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

        history = modeling(model)

        val_loss = history.history["val_loss"][-1]

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_rad = rad

    hp["radius"] = min_rad
    min_val_loss = 10000

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

        history = modeling(model)

        val_loss = history.history["val_loss"][-1]

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            min_leaky = elem

    hp["leaky"] = min_leaky
    min_val_loss = 10000

    saved_model = None

    # layers, units, concat
    """for n_layers in layers:
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

                history = modeling(model)

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
    """

    for elem in units:
        tf.keras.backend.clear_session()
        model = Sequential()

        model.add(SimpleDeepReservoirLayer(input_shape=(None, channel.X_train.shape[2]),
                                           units=elem,
                                           return_sequences=hp["return_sequences"],
                                           layers=hp["layers"],
                                           concat=hp["concat"],
                                           input_scaling=hp["input_scaling"],
                                           inter_scaling=hp["inter_scaling"],
                                           spectral_radius=hp["radius"],
                                           leaky=hp["leaky"],
                                           connectivity_recurrent=hp["connectivity_recurrent"],
                                           connectivity_inter=hp["connectivity_inter"],
                                           connectivity_input=hp["connectivity_input"]
                                           )
                  )

        history = modeling(model)

        val_loss = history.history["val_loss"][-1]

        if val_loss <= min_val_loss:
            min_val_loss = val_loss
            reservoir_units = elem
            saved_model = model

    hp["units"] = reservoir_units
    saved_model.save_weights(os.path.join("/hp/{}/weights", '{}_weights.h5'.format(data,channel)))


    """
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

            history = modeling(model)

            val_loss = history.history["val_loss"][-1]

            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                min_inter_scaling = elem
                save_model = model

        hp["inter_scaling"] = min_inter_scaling
        saved_model.save_weights(os.path.join("/hp/weights", '{}_weights.h5'.format(channel)))
        """

    hp["validation_loss"] = min_val_loss

    f = open(os.path.join('hp/config',data, '{}.yaml'.format(row.chan_id)), "w")

    yaml.dump(hp, f, default_flow_style=False)
    f.close()
