import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class cust():

    def __init__(self, dat):
        self.dat = dat

    def xy(self, x_dummies_list, X_list, y_list):
        X = self.dat[X_list]
        y = self.dat[y_list]

        X = pd.get_dummies(X, columns=x_dummies_list, drop_first=True)

        x_cols = X.columns

        X = X.values
        y = y.values

        y = np.ravel(y)

        return X, y, x_cols

    def xy_df(self, x_dummies_list, X_list, y_list):
        X = self.dat[X_list]
        y = self.dat[y_list]

        X = pd.get_dummies(X, columns=x_dummies_list, drop_first=True)

        x_cols = X.columns

        y = np.ravel(y)

        return X, y, x_cols

    def clean_data(self):
        self.dat.replace([np.inf, -np.inf], np.nan)
        self.dat = self.dat.dropna(axis=0, how='any')

        self.dat = self.dat[~self.dat.isin([np.nan, np.inf, -np.inf]).any(1)]
        return self.dat

    def outlier_removal(self, var):
        IQR = self.dat[var].describe()['75%'] - self.dat[var].describe()['25%']
        min_val = self.dat[var].describe()['25%'] - (IQR * 1.5)
        max_val = self.dat[var].describe()['75%'] + (IQR * 1.5)

        self.dat = self.dat[(self.dat[var] > min_val) & (self.dat[var] < max_val)]
        plt.boxplot(self.dat[var])
        return self.dat

    @staticmethod
    def comparison_df(y_pred, y_test):
        comparison_df = pd.DataFrame({'y_pred': y_pred, 'y_test': y_test})
        comparison_df['abs_difference'] = abs(comparison_df['y_pred'] - comparison_df['y_test'])
        comparison_df['real_difference'] = comparison_df['y_pred'] - comparison_df['y_test']
        print(comparison_df.describe())

        print(comparison_df.sum())

        print("Average Difference: ", comparison_df.sum()[2] / len(comparison_df))

        return comparison_df


import numpy as np
import matplotlib.pyplot as plt


def create_series(df, xcol, datecol):
    features_considered = [xcol]
    features = df[features_considered]
    features.index = df[datecol]
    features.head()
    features.plot(subplots=True)
    return features


def stationarity_test(X, log_x="Y", return_p=False, print_res=True):
    if log_x == "Y":
        X = np.log(X[X > 0])

    from statsmodels.tsa.stattools import adfuller
    dickey_fuller = adfuller(X)

    if print_res:

        print('ADF Stat is: {}.'.format(dickey_fuller[0]))

        print('P Val is: {}.'.format(dickey_fuller[1]))
        print('Critical Values (Significance Levels): ')
        for key, val in dickey_fuller[4].items():
            print(key, ":", round(val, 3))

    if return_p:
        return dickey_fuller[1]


def difference(X):
    diff = X.diff()
    plt.plot(diff)
    plt.show()
    return diff


import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import pandas as pd


class Data_Prep:

    def __init__(self, dataset):
        self.dataset = dataset

    def preprocess_rnn(self, date_colname, numeric_colname, pred_set_timesteps):
        features = (create_series(self.dataset, numeric_colname, date_colname)).sort_index()
        rnn_df = features.groupby(features.index).sum()

        timestep_idx = len(rnn_df) - pred_set_timesteps
        validation_df = rnn_df.iloc[timestep_idx:]
        rnn_df = rnn_df.iloc[1:timestep_idx, ]

        print("Summary Statistics - ADF Test For Stationarity\n")
        if stationarity_test(X=rnn_df[numeric_colname], return_p=True, print_res=False) > 0.05:
            print("P Value is high. Consider Differencing: " + str(stationarity_test(X=rnn_df[numeric_colname], return_p=True, print_res=False)))
        else:
            stationarity_test(X=rnn_df[numeric_colname])

        rnn_df = rnn_df.sort_index(ascending=True)
        rnn_df = rnn_df.reset_index()

        return rnn_df, validation_df


class Series_Prep:

    def __init__(self, rnn_df, numeric_colname):
        self.rnn_df = rnn_df
        self.numeric_colname = numeric_colname

    def make_window(self, sequence_length, train_test_split, return_original_x=True):

        result = []

        for index in range(len(self.rnn_df) - sequence_length):
            result.append(self.rnn_df[self.numeric_colname][index: index + sequence_length])

        train_test_split = 0.9
        row = int(round(train_test_split * np.array(result).shape[0]))
        train = np.array(result)[:row, :]
        X_train = train[:, :-1]

        X_min = X_train.min()
        X_max = X_train.max()

        X_min_orig = X_train.min()
        X_max_orig = X_train.max()

        def minmax(X):
            return (X - X_min) / (X_max - X_min)

        def reverse_minmax(X):
            return X * (X_max - X_min) + X_min

        def minmax_windows(window_data):
            normalised_data = []
            for window in window_data:
                window.index = range(sequence_length)
                normalised_window = [((minmax(p))) for p in window]
                normalised_data.append(normalised_window)
            return normalised_data

        result = minmax_windows(result)

        result = np.array(result)
        if return_original_x:
            return result, X_min_orig, X_max_orig
        else:
            return result

    @staticmethod
    def reshape_window(window, train_test_split=0.8):

        row = round(train_test_split * window.shape[0])
        train = window[:row, :]

        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = window[row:, :-1]
        y_test = window[row:, -1]

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_train = np.reshape(y_train, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))

        return X_train, X_test, y_train, y_test


class Predict_Future:

    def __init__(self, X_test, validation_df, lstm_model):
        self.X_test = X_test
        self.validation_df = validation_df
        self.lstm_model = lstm_model

    def predicted_vs_actual(self, X_min, X_max, numeric_colname):

        curr_frame = self.X_test[len(self.X_test) - 1]
        future = []

        for i in range(len(self.validation_df)):
            future.append(self.lstm_model.predict(curr_frame[newaxis, :, :])[0, 0])

            curr_frame = np.insert(curr_frame, len(self.X_test[0]), future[-1], axis=0)

            curr_frame = curr_frame[1:]

        def reverse_minmax(X, X_max=X_max, X_min=X_min):
            return X * (X_max - X_min) + X_min

        reverse_curr_frame = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in self.X_test[len(self.X_test) - 1]],
                                           "historical_flag": 1})
        reverse_future = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in future],
                                       "historical_flag": 0})

        print("See Plot for predicted vs. actuals")
        plt.plot(reverse_curr_frame[numeric_colname])
        plt.plot(reverse_future[numeric_colname])
        plt.title("Predicted Points Vs. Actuals (Validation)")
        plt.show()

        comparison_df = pd.DataFrame({"Validation": self.validation_df[numeric_colname],
                                      "Predicted": [reverse_minmax(x) for x in future]})
        print("Validation Vs. Predicted")
        print(comparison_df.sum())

    def predict_future(self, X_min, X_max, numeric_colname, timesteps_to_predict, return_future=True):

        curr_frame = self.X_test[len(self.X_test) - 1]
        future = []

        for i in range(timesteps_to_predict):
            future.append(self.lstm_model.predict(curr_frame[newaxis, :, :])[0, 0])

            curr_frame = np.insert(curr_frame, len(self.X_test[0]), future[-1], axis=0)

            curr_frame = curr_frame[1:]

        def reverse_minmax(X, X_max=X_max, X_min=X_min):
            return X * (X_max - X_min) + X_min

        reverse_curr_frame = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in self.X_test[len(self.X_test) - 1]],
                                           "historical_flag": 1})
        reverse_future = pd.DataFrame({numeric_colname: [reverse_minmax(x) for x in future],
                                       "historical_flag": 0})

        reverse_future.index += len(reverse_curr_frame)

        print("See Plot for Future Predictions")
        plt.plot(reverse_curr_frame[numeric_colname])
        plt.plot(reverse_future[numeric_colname])
        plt.title("Predicted Future of " + str(timesteps_to_predict) + " days")
        plt.show()

        if return_future:
            return reverse_future


import os
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

from google.colab import drive

drive.mount("/drive/")

dat = pd.read_csv('/drive/MyDrive/owid-co2-data.csv')

dat = dat[dat["country"] == "United States"]
max1 = max(dat["co2"].values)
dat["co2"] /= max1

split = 0.8
sequence_length = 48

data_prep = Data_Prep(dataset=dat)
rnn_df, validation_df = data_prep.preprocess_rnn(date_colname='year', numeric_colname='co2', pred_set_timesteps=16)

series_prep = Series_Prep(rnn_df=rnn_df, numeric_colname='co2')
window, X_min, X_max = series_prep.make_window(sequence_length=sequence_length,
                                               train_test_split=split,
                                               return_original_x=True)

X_train, X_test, y_train, y_test = series_prep.reshape_window(window, train_test_split=split)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Bidirectional, Dropout, Input
from keras.callbacks import ReduceLROnPlateau  # Learning rate scheduler for when we reach plateaus


def reset_weights(model):
    import keras.backend as K
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
        if hasattr(layer, 'bias_initializer'):
            layer.bias.initializer.run(session=session)


EPOCHS = 50
validation = 0.05

model = Sequential()

# model.add(LSTM(
# input_shape = (sequence_length-1, 1), activation='relu', return_sequences = True,
# units = 256))

model.add(Input(shape=(sequence_length - 1, 1)))

model.add(Conv1D(filters=128, kernel_size=3,
                 strides=1, padding="same",
                 activation="relu"))

model.add(LSTM(activation='relu', return_sequences=False, units=128))

# model.add(Conv1D(16, 32, padding = "same", activation='relu', input_shape = (sequence_length-1, 1)))

# model.add(LSTM(128,return_sequences=False))


# model.add(LSTM(


model.add(Dropout(0.15))

model.add(Dense(
    units=32, activation='relu'))

model.add(Dense(
    units=16, activation="linear"))

model.compile(loss='mse', optimizer='adam')

history = model.fit(X_train, y_train, epochs=EPOCHS, validation_split=validation)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

future = Predict_Future(X_test=X_test, validation_df=validation_df, lstm_model=model)

future.predicted_vs_actual(X_min=X_min, X_max=X_max, numeric_colname='co2')

future.predict_future(X_min=X_min, X_max=X_max, numeric_colname='co2', timesteps_to_predict=128, return_future=True)

dat

data = pd.read_csv('/drive/MyDrive/owid-co2-data.csv', index_col="year")

data = data[data["country"] == "United States"]

da = data[['co2']]

da.plot(color="green")

import numpy as np


def sampling(sequence, n_steps):
    X, Y = list(), list()

    for i in range(len(sequence)):

        sam = i + n_steps
        if sam > len(sequence) - 1:
            break

        x, y = sequence[i:sam], sequence[sam]

        X.append(x)
        Y.append(y)
        return np.array(X), np.array(Y)


n_steps = 18

X, Y = sampling(da['co2'].tolist(), n_steps)

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

X = X.reshape((X.shape[0], X.shape[1], 1))

model.fit(X, Y, epochs=200, verbose=0)

x = X[-1]

x.shape
x = x.reshape((1, n_steps, 1))

ypred = model.predict(x, verbose=0)

ypred

