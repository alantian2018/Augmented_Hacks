import pandas as pd
import numpy as np
import json

df = pd.read_csv('owid-co2-data.csv')
df.sort_values(by=['year'], inplace=True)

outfile = open('co2data.json', 'w')
co2preds = {}

for country in df['country'].unique():
    data = df[df['country'] == country]['co2'].dropna().values
    data = np.array(data)
    if len(data) > 3:
        poly = np.polyfit(np.arange(len(data)), data, 5)
        print(country, poly, sep=",")
        preds = []
        for i in range(100):
            preds.append(np.polyval(poly, 2022+i))
        co2preds[country] = preds

json.dump(co2preds, outfile)