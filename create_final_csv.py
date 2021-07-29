import pandas as pd

columns = ["Model", "Precision", "Recall", "True Positives", "False Positives", "False Negatives", "Training Time", "Params"]

lstm_row = ["LSTM", "0.80", "0.78", "82", "20", "23", "0h:48m:09s", "86,250"]
esn_row = ["ESN", "0.80", "0.70", "74", "19", "31", "0h:21m:55s", "2,000-9,000"]

rows = [lstm_row, esn_row]

df = pd.DataFrame(columns=columns)
df.append(lstm_row)
for i in range(len(rows)):
    df.loc[i] = rows[i]

df.to_csv('./data/final.csv', sep=',')