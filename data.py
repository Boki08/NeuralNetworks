import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


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
            self.test = pd.read_csv(test_file, sep=';')
        except OSError:
            print(test_file + "does not exist")
            self.imported = False
            return
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported = False
            raise

        self.test_data_copy = self.test.copy()
        self.test_copy = self.test.copy()

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

        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def split_sequences(self, sequences_x, sequences_y, n_steps):
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

    def importFiles(self):

        unique_values_src_ip = self.train_data['Source'].unique()
        unique_values_dst_ip = self.train_data['Destination'].unique()
        unique_values_src_ip_t = self.test['Source'].unique()
        unique_values_dst_ip_t = self.test['Destination'].unique()

        self.unique_values_normal_attack = self.train_data['Normal/Attack'].unique()

        unique_values_protocol_train = self.train_data['Protocol'].unique()
        unique_values_protocol_test = self.test['Protocol'].unique()

        unique_values_ips = np.concatenate((unique_values_src_ip, unique_values_src_ip_t, unique_values_dst_ip,
                                            unique_values_dst_ip_t), axis=None)

        unique_values_ips = unique_values_ips[np.sort(np.unique(unique_values_ips, return_index=True)[1])]

        unique_values_protocol = np.concatenate((self.train_data['Protocol'].unique(), self.test['Protocol'].unique()),
                                                axis=None)

        unique_values_protocol = unique_values_protocol[
            np.sort(np.unique(unique_values_protocol, return_index=True)[1])]

        for idx, val in enumerate(unique_values_ips):
            if val in unique_values_src_ip:
                self.train_data['Source'] = self.train_data['Source'].replace({val: idx})
            if val in unique_values_dst_ip:
                self.train_data['Destination'] = self.train_data['Destination'].replace({val: idx})
            if val in unique_values_src_ip_t:
                self.test['Source'] = self.test['Source'].replace({val: idx})
            if val in unique_values_dst_ip_t:
                self.test['Destination'] = self.test['Destination'].replace({val: idx})

        for idx, val in enumerate(unique_values_protocol):
            if val in unique_values_protocol_train:
                self.train_data['Protocol'] = self.train_data['Protocol'].replace({val: idx})
            if val in unique_values_protocol_test:
                self.test['Protocol'] = self.test['Protocol'].replace({val: idx})

        for idx, val in enumerate(self.unique_values_normal_attack):
            self.train_data['Normal/Attack'] = self.train_data['Normal/Attack'].replace({val: idx})

        cols = [i for i in self.train_data.columns if
                i not in ["Timestamp", "Time", "Normal/Attack", "Flow ID", "Label", "Info", "Encapsulation type",
                          "No."]]
        for col in cols:
            if self.train_data[col].isna().values.any():
                self.train_data[col].fillna(0, inplace=True)
            if self.train_data[col].dtype == "object":
                if ~any(self.train_data[col].str.contains("nan", regex=False)) & (
                        any(self.train_data[col].str.contains("Active", regex=False))
                        | any(self.train_data[col].str.contains("Inactive", regex=False))):
                    self.train_data[col] = self.train_data[col].replace({'Active': 2, 'Inactive': 1}).astype(float)
                else:
                    self.train_data[col] = self.train_data[col].str.replace(',', '.').astype(float)
            else:
                self.train_data[col] = self.train_data[col].astype(float)

        if self.train_data.isna().values.any():
            self.train_data.fillna(0, inplace=True)

        if self.test.isna().values.any():
            self.test.fillna(0, inplace=True)

        cols = [i for i in self.test.columns if
                i in ["Source", "Destination", "Protocol", "Length", "Source Port", "Destination Port"]]
        for col in cols:
            if self.test[col].isna().values.any():
                self.test[col].fillna(0, inplace=True)
            if self.test[col].dtype == "object":
                if ~any(self.test[col].str.contains("nan", regex=False)) & (
                        any(self.test[col].str.contains("Active", regex=False))
                        | any(
                    self.test[col].str.contains("Inactive", regex=False))):
                    self.test[col] = self.test[col].replace({'Active': 2, 'Inactive': 1}).astype(float)
                else:
                    self.test[col] = self.test[col].str.replace(',', '.').astype(float)
            else:
                self.test[col] = self.test[col].astype(float)

        if self.test.isna().values.any():
            self.test.fillna(0, inplace=True)

        self.test.replace([np.inf, -np.inf], 0, inplace=True)

        self.train_data = self.train_data.drop(self.train_data.columns.difference(
            ["Source", "Destination", "Protocol", "Length", "Source Port", "Destination Port", "Normal/Attack"]), 1)

        self.x_test = self.test.drop(self.test.columns.difference(
            ["Source", "Destination", "Protocol", "Length", "Source Port", "Destination Port"]), 1)

        self.train_data.reset_index(drop=True, inplace=True)

        self.x_train = self.train_data.iloc[:, :-1]
        self.y_train = self.train_data.iloc[:, -1]

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(self.x_train)
        self.x_train = scaler.transform(self.x_train)
        self.x_train_scaled = self.x_train

        self.x_test = scaler.transform(self.x_test)
        self.x_test_scaled = self.x_test.copy()

        self.y_train_full = self.y_train.copy()

        self.x_train, self.y_train = self.split_sequences(self.x_train, self.y_train, self.n_steps)

        self.n_features = self.x_train.shape[2]

        print(self.x_train.shape, self.y_train.shape)

    def split_data(self, type_of_split):
        self.type_of_split = type_of_split
        if type_of_split == "Testing":

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

            self.x_train, self.y_train = self.split_sequences(self.x_train, self.y_train, self.n_steps)

            self.n_features = self.x_train.shape[2]

            print(self.x_train.shape, self.y_train.shape)
        else:
            self.x_train = self.x_train_scaled.copy()
            self.x_test = self.x_test_scaled.copy()
            self.y_train = self.y_train_full.copy()
            self.test_copy = self.test_data_copy.copy()

            self.x_train, self.y_train = self.split_sequences(self.x_train, self.y_train, self.n_steps)

            self.n_features = self.x_train.shape[2]

            print(self.x_train.shape, self.y_train.shape)
