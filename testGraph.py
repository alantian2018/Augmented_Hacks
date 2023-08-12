import pandas as pd
import numpy as np
import json
from matplotlib import pyplot as plt

df = pd.read_csv('owid-co2-data.csv')
df.sort_values(by=['year'], inplace=True)

outfile = open('co2data.json', 'w')
co2preds = {}

for country in df['country'].unique():
    data2 = df[df['country'] == country]
    data = df[df['country'] == country]['co2'].dropna().values
    data = np.array(data)
    if len(data) > 3:
        y = []
        x = []
        for v in data2[['year', 'co2']].values:
            x.append(v[0])
            y.append(v[1])
        poly = np.polyfit(x, y, 3)
        # print(country, poly, sep=",")
        preds = []
        for i in range(100):
            preds.append(np.polyval(poly, 2022 + i))
        co2preds[country] = str(preds)

        if country == "United States":
            for v in data2[['year', 'co2']].values:
                plt.scatter(v[0], v[1], color='red')
                plt.scatter(v[0], np.polyval(poly, v[0]), color='blue')
            for v in range(50):
                plt.scatter(2022 + v, abs(np.polyval(poly, 2022 + v)), color='green')
plt.show()
json.dump(co2preds, outfile)
