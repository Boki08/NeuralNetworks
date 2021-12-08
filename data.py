from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


class Data:
    def __init__(self, train_file, test_file):

        self.imported = True

        try:
            self.train_data = pd.read_csv(train_file, sep=';')
        except OSError:
            print(train_file + "does not exist")
            self.imported = False
            return
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported = False
            raise

        self.train_data_copy = self.train_data.copy()

        try:
            self.test_data = pd.read_csv(test_file, sep=';')
        except OSError:
            print(test_file + "does not exist")
            self.imported = False
            return
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported = False
            raise

        self.test_data_copy = self.test_data.copy()
        self.test_copy = self.test_data.copy()

        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []

        self.y_train_full = []

        self.n_steps = 1
        self.n_features = 0

        self.unique_values_normal_attack = {}

        self.type_of_split = "Regular"

        self.x_train_scaled = []
        self.x_test_scaled = []

        self.columns = ["Source", "Destination", "Protocol", "Length", "Source Port", "Destination Port"]

    def importFiles(self):
        unique_values_train = {}
        unique_values_test = {}
        unique_values = {}
        cols = [i for i in self.train_data.columns if i in self.columns]
        for col in cols:
            if self.train_data[col].isna().values.any():
                self.train_data[col].fillna(-1, inplace=True)
            if self.train_data[col].dtype == "object":
                unique_values_train[col] = self.train_data[col].unique()
            if self.test_data[col].isna().values.any():
                self.test_data[col].fillna(-1, inplace=True)
            if self.test_data[col].dtype == "object":
                unique_values_test[col] = self.test_data[col].unique()

        self.unique_values_normal_attack = self.train_data['Normal/Attack'].unique()

        for idx, val in enumerate(self.unique_values_normal_attack):
            self.train_data['Normal/Attack'] = self.train_data['Normal/Attack'].replace({val: idx})

        for key in unique_values_train:
            unique_values[key] = np.hstack([unique_values_train[key], unique_values_test[key]])
            unique_values[key] = unique_values[key][np.sort(np.unique(unique_values[key], return_index=True)[1])]
            for idx, val in enumerate(unique_values[key]):
                if val in unique_values_train[key]:
                    self.train_data = self.train_data.replace({val: idx})
                if val in unique_values_test[key]:
                    self.test_data = self.test_data.replace({val: idx})

        self.train_data = self.train_data.drop(
            columns=self.train_data.columns.difference(self.columns + ["Normal/Attack"]))

        self.train_data.reset_index(drop=True, inplace=True)

        self.x_train = self.train_data.iloc[:, :-1]
        self.y_train = self.train_data.iloc[:, -1]

        self.x_test = self.test_data.drop(columns=self.test_data.columns.difference(self.columns))

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_train_scaled = self.x_train

        self.x_test = scaler.transform(self.x_test)
        self.x_test_scaled = self.x_test.copy()

        self.y_train_full = self.y_train.copy()

        self.x_train, self.y_train = split_sequences(self.x_train, self.y_train, self.n_steps)

        self.n_features = self.x_train.shape[2]

        print("\tData shape set to regular")
        print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape), '{}'.format(self.y_train.shape)))

    def split_data(self, type_of_split):
        self.type_of_split = type_of_split
        if type_of_split == "Training":

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.train_data.iloc[:, :-1], self.train_data.iloc[:, -1],
                test_size=0.2, random_state=0)

            self.test_copy = self.train_data_copy.loc[self.x_test.index.tolist()]

            self.x_train = pd.DataFrame(self.x_train).reset_index(drop=True)
            self.x_train = self.x_train.to_numpy()
            self.x_test = pd.DataFrame(self.x_test).reset_index(drop=True)
            self.x_test = self.x_test.to_numpy()
            self.y_train = pd.DataFrame(self.y_train).reset_index(drop=True)
            self.y_train = self.y_train.to_numpy()

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(self.x_train)
            self.x_train = scaler.transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)

            self.x_train, self.y_train = split_sequences(self.x_train, self.y_train, self.n_steps)

            self.n_features = self.x_train.shape[2]

            print("\tData shape set to training")
            print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape),
                                                         '{}'.format(self.y_train.shape)))
            print("\tx_test: {:s}, y_test:{:s}".format('{}'.format(self.x_test.shape), '{}'.format(self.y_test.shape)))
        else:
            self.x_train = self.x_train_scaled.copy()
            self.x_test = self.x_test_scaled.copy()
            self.y_train = self.y_train_full.copy()
            self.test_copy = self.test_data_copy.copy()

            self.x_train, self.y_train = split_sequences(self.x_train, self.y_train, self.n_steps)

            self.n_features = self.x_train.shape[2]

            print("\tData shape set to regular")
            print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape),
                                                         '{}'.format(self.y_train.shape)))


def split_sequences(sequences_x, sequences_y, n_steps):
    x, y = list(), list()
    for i in range(len(sequences_x)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences_x):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences_x[i:end_ix, :], sequences_y[end_ix - 1]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)