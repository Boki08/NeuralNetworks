from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
import pandas as pd
import numpy as np

class MyLabelEncoder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        self.classes_ = pd.Series(y).unique()
        return self

class Data:
    def __init__(self, train_file, test_file):

        self.imported_train = True
        self.imported_test = True

        try:
            self.train_data = pd.read_csv(train_file, sep=';')
        except OSError:
            print(train_file + " does not exist")
            self.imported_train = False
            return
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported_train = False
            raise

        self.train_data_copy = self.train_data.copy()

        try:
            self.test_data = pd.read_csv(test_file, sep=';')
        except OSError:
            print(test_file + " does not exist")
            self.imported_test = False
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            self.imported_test = False
            raise

        if self.imported_test:
            self.test_data_copy = self.test_data.copy()

        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.x_val = []
        self.y_val = []

        self.y_train_full = []

        self.n_steps = 1
        self.n_features = 0

        self.unique_values_normal_attack = {}

        self.type_of_split = "Train"

        self.x_train_scaled = []
        self.x_test_scaled = []
        self.test_copy = []

        self.columns = ["Source", "Destination", "Protocol", "Length", "Source Port", "Destination Port"]

        unique_values = []
        unique_values_test = []
        object_columns = []

        cols = [i for i in self.train_data.columns if i in self.columns]
        for col in cols:
            if self.train_data[col].isna().values.any():
                self.train_data[col].fillna(-1, inplace=True)
            if self.train_data[col].dtype == "object":
                object_columns.append(col)
                unique_values = [*unique_values, *self.train_data[col].unique()]
            if self.imported_test:
                if self.test_data[col].isna().values.any():
                    self.test_data[col].fillna(-1, inplace=True)
                if self.test_data[col].dtype == "object":
                    unique_values_test = [*unique_values_test, *self.test_data[col].unique()]

        if self.imported_test:
            unique_values = [*unique_values, *unique_values_test]

        label_encoder = MyLabelEncoder()
        label_encoder.fit(unique_values)
        for key in object_columns:
            self.train_data[key] = label_encoder.transform(self.train_data[key])
            if self.imported_test:
                self.test_data[key] = label_encoder.transform(self.test_data[key])

        self.unique_values_normal_attack = self.train_data['Normal/Attack'].unique()

        self.unique_values_normal_attack[[0, np.where(self.unique_values_normal_attack == "Attack")[0][0]]] = \
            self.unique_values_normal_attack[[np.where(self.unique_values_normal_attack == "Attack")[0][0], 0]]

        for idx, val in enumerate(self.unique_values_normal_attack):
            self.train_data['Normal/Attack'] = self.train_data['Normal/Attack'].replace({val: idx})

        self.train_data = self.train_data.drop(
            columns=self.train_data.columns.difference(self.columns + ["Normal/Attack"]))

        self.train_data.reset_index(drop=True, inplace=True)

        self.y_train_copy = self.train_data["Normal/Attack"].to_numpy()

        self.x_train_copy = self.train_data.drop(columns=self.train_data.columns.difference(self.columns))

        # normalize the dataset
        if self.imported_test:
            self.x_test = self.test_data.drop(columns=self.test_data.columns.difference(self.columns))

            scaler = MinMaxScaler(feature_range=(0, 1))
            self.x_train_scaled = scaler.fit_transform(self.x_train_copy)
            self.x_test_scaled = scaler.transform(self.x_test)

    def split_data(self, type_of_split):
        self.type_of_split = type_of_split
        if type_of_split == "Train":

            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_train_copy, self.y_train_copy,
                test_size=0.4, random_state=None)

            self.x_test, self.x_val, self.y_test, self.y_val = train_test_split(self.x_test,
                                                                                self.y_test,
                                                                                test_size=0.5, random_state=None)

            self.test_copy = self.train_data_copy.loc[self.x_test.index.tolist()]

            self.x_train = pd.DataFrame(self.x_train).reset_index(drop=True).to_numpy()
            self.y_train = pd.DataFrame(self.y_train).reset_index(drop=True).to_numpy()
            self.x_test = pd.DataFrame(self.x_test).reset_index(drop=True).to_numpy()
            self.y_test = pd.DataFrame(self.y_test).reset_index(drop=True).to_numpy()
            self.x_val = pd.DataFrame(self.x_val).reset_index(drop=True).to_numpy()
            self.y_val = pd.DataFrame(self.y_val).reset_index(drop=True).to_numpy()

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(self.x_train)
            self.x_train = scaler.transform(self.x_train)
            self.x_test = scaler.transform(self.x_test)
            self.x_val = scaler.transform(self.x_val)

            self.x_train, self.y_train = split_sequences(self.x_train, self.y_train, self.n_steps)

            self.n_features = self.x_train.shape[2]

            print("\n\tData shape set to train")
            print("\tx_train: {:s}, y_train:{:s}".format('{}'.format(self.x_train.shape),
                                                         '{}'.format(self.y_train.shape)))
            print("\tx_test: {:s}, y_test:{:s}".format('{}'.format(self.x_test.shape), '{}'.format(self.y_test.shape)))
            print("\tx_val: {:s}, y_val:{:s}".format('{}'.format(self.x_val.shape), '{}'.format(self.y_val.shape)))
        else:
            self.x_train = self.x_train_scaled.copy()
            self.x_test = self.x_test_scaled.copy()
            self.y_train = self.y_train_copy.copy()
            self.test_copy = self.test_data_copy.copy()

            self.x_train, self.y_train = split_sequences(self.x_train, self.y_train, self.n_steps)

            self.n_features = self.x_train.shape[2]

            print("\n\tData shape set to regular")
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