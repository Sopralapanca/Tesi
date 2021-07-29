import pandas as pd
from telemanom.helpers import Config
from telemanom.channel import Channel
from telemanom.utility import create_esn_model, create_lstm_model
import yaml

config = Config("./config.yaml")
chan_df = pd.read_csv("./labeled_anomalies.csv")

array = []

for i, row in chan_df.iterrows():
    channel = Channel(config, row.chan_id)
    channel.load_data()

    path =("hp/2021-07-27_12.50.38/config/{}.yaml".format(row.chan_id))
    with open(path, 'r') as file:
        hp = yaml.load(file, Loader=yaml.BaseLoader)

    units = int(hp["units"])
    n_predictions = config.n_predictions

    trainable_params= units*n_predictions
    array.append(trainable_params)

min = min(array)
max = max(array)

with open("./data/esnmmodelsummary.txt", 'w') as f:
    f.write("total params\n")
    f.write("min: {}\n".format(str(min)))
    f.write("max: {}\n".format(str(max)))

channel = Channel(config, "A-1")
channel.load_data()

model = create_lstm_model(channel, config)

from contextlib import redirect_stdout

with open('./data/lstmmodelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

