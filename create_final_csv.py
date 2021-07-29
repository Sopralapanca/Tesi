import pandas as pd

columns = ["Model", "Precision", "Recall", "True Positives", "False Positives", "False Negatives", "Training Time", "Params"]

lstm_row = ["LSTM", "0.80", "0.78", "82", "20", "23", "0h:48m:09s", "86,250"]
deep_esn_row = ["DeepESN", "0.74", "0.61", "64", "23", "41", "5h:22m:28s", ""]
esn_row = ["ESN", "0.80", "0.70", "74", "19", "31", "0h:21m:55s", "2,000-9,000"]

rows = [lstm_row, deep_esn_row, esn_row]

df = pd.DataFrame(columns=columns)
df.append(lstm_row)
for i in range(len(rows)):
    df.loc[i] = rows[i]

df.to_csv('./data/final.csv', sep=',')