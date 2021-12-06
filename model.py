import keras.models
import keras.layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import initializers
from keras_self_attention import SeqSelfAttention
from sklearn.metrics import confusion_matrix

class Model:
    def __init__(self):
        self.kerasModel = keras.models.Sequential()
        self.model_name = "model"
        self.models = {}

    def train_val_loss(self, history):
        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        plt.plot(loss_train, 'y', label='Training loss')
        plt.plot(loss_val, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def train_val_accuracy(self, history):
        loss_train = history.history['accuracy']
        loss_val = history.history['val_accuracy']
        plt.plot(loss_train, 'g', label='Training accuracy')
        plt.plot(loss_val, 'b', label='Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    def createClassicLSTMModel(self, x, y):
        ######################################################## classic
        self.kerasModel = keras.models.Sequential()
        lstm_out = 196
        self.kerasModel.add(keras.layers.LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.1))
        self.kerasModel.add(keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = self.kerasModel.fit(x, y, batch_size=128, epochs=25, validation_split=0.1, verbose=1)

        print(self.kerasModel.summary())
        self.train_val_loss(history)
        self.train_val_accuracy(history)

        self.model_name = "ClassicLSTMModel"
        self.models[self.model_name] = self.kerasModel
        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}".format(self.model_name))
        ########################################################

    def createStackedLSTMModel(self, x, y):
        ######################################################## stacked
        self.kerasModel = keras.models.Sequential()
        lstm_out = 196
        self.kerasModel.add(
            keras.layers.LSTM(lstm_out, input_shape=(1, 6), recurrent_dropout=0.1, return_sequences=True))
        self.kerasModel.add(keras.layers.Dropout(0.1))
        self.kerasModel.add(
            keras.layers.LSTM(lstm_out, activation='relu',
                              recurrent_dropout=0.1, return_sequences=True))
        self.kerasModel.add(keras.layers.Dropout(0.1))
        self.kerasModel.add(
            keras.layers.LSTM(lstm_out, activation='relu',
                              recurrent_dropout=0.1, return_sequences=True))
        self.kerasModel.add(keras.layers.Dropout(0.1))
        self.kerasModel.add(
            keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = self.kerasModel.fit(x, y, epochs=25, batch_size=128, validation_split=0.1, verbose=1)

        print(self.kerasModel.summary())
        self.train_val_loss(history)
        self.train_val_accuracy(history)

        self.model_name = "StackedLSTMModel"
        self.models[self.model_name] = self.kerasModel
        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}".format(self.model_name))
        ########################################################

    def createBidirectionalLSTMModel(self, x, y):
        ######################################################## Bidirectional
        self.kerasModel = keras.models.Sequential()
        lstm_out = 64
        self.kerasModel.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_out, return_sequences=True)))
        self.kerasModel.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_out)))
        self.kerasModel.add(keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = self.kerasModel.fit(x, y, epochs=50, batch_size=128, validation_split=0.1, verbose=1)

        print(self.kerasModel.summary())
        self.train_val_loss(history)
        self.train_val_accuracy(history)

        self.model_name = "BidirectionalLSTMModel"
        self.models[self.model_name] = self.kerasModel
        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}".format(self.model_name))
        ########################################################

    def createGRUModel(self, x, y):
        ######################################################## GRU(Gated Recurrent Unit)
        self.kerasModel = keras.models.Sequential()
        gru_out = 25
        self.kerasModel.add(keras.layers.GRU(gru_out, input_shape=(1, 6), return_sequences=True))
        self.kerasModel.add(keras.layers.GRU(gru_out, activation='relu', return_sequences=True))
        self.kerasModel.add(keras.layers.Dropout(0.5))
        self.kerasModel.add(SeqSelfAttention())
        self.kerasModel.add(keras.layers.GRU(gru_out, return_sequences=False))
        self.kerasModel.add(keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


        history = self.kerasModel.fit(x, y, epochs=20, batch_size=25, validation_split=0.1, verbose=1)

        print(self.kerasModel.summary())
        self.train_val_loss(history)
        self.train_val_accuracy(history)

        self.model_name = "GRUModel"
        self.models[self.model_name] = self.kerasModel
        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}".format(self.model_name))
        ########################################################

    def loadClassicLSTMModel(self):
        ######################################################## classic
        temp_name = self.model_name
        self.model_name = "ClassicLSTMModel"
        if self.model_name not in self.models:
            try:
                self.kerasModel = keras.models.load_model('ClassicLSTMModel')
                self.model_name = "ClassicLSTMModel"
                self.models[self.model_name] = self.kerasModel

                print("\tLoaded {:s}".format(self.model_name))
            except OSError:
                print("Classic Model does not exist")
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        else:
            self.model_name = temp_name
        ########################################################

    def loadStackedLSTMModel(self):
        ######################################################## stacked
        temp_name = self.model_name
        self.model_name = "StackedLSTMModel"
        if self.model_name not in self.models:
            try:
                self.kerasModel = keras.models.load_model('StackedLSTMModel')
                self.model_name = "StackedLSTMModel"
                self.models[self.model_name] = self.kerasModel

                print("\tLoaded {:s}".format(self.model_name))
            except OSError:
                print("Stacked Model does not exist")
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        else:
            self.model_name = temp_name
        ########################################################

    def loadBidirectionalLSTMModel(self):
        ######################################################## Bidirectional
        temp_name = self.model_name
        self.model_name = "BidirectionalLSTMModel"
        if self.model_name not in self.models:
            try:
                self.kerasModel = keras.models.load_model('BidirectionalLSTMModel')
                self.model_name = "BidirectionalLSTMModel"
                self.models[self.model_name] = self.kerasModel

                print("\tLoaded {:s}".format(self.model_name))
            except OSError:
                print("Bidirectional Model does not exist")
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        else:
            self.model_name = temp_name
        ########################################################

    def loadGRUModel(self):
        ######################################################## GRU(Gated Recurrent Unit)
        temp_name = self.model_name
        self.model_name = "GRUModel"
        if self.model_name not in self.models:
            try:
                self.kerasModel = keras.models.load_model('GRUModel')
                self.model_name = "GRUModel"
                self.models[self.model_name] = self.kerasModel

                print("\tLoaded {:s}".format(self.model_name))
            except OSError:
                print("GRU Model does not exist")
            except BaseException as err:
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        else:
            self.model_name = temp_name
        ########################################################

    def simulate(self, imported_data):
        x_input = np.array(imported_data.x_test)
        x_input = x_input.reshape((x_input.shape[0], imported_data.n_steps, imported_data.n_features))

        x_pred = self.models[self.model_name].predict(x_input)
        x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1])

        preds = imported_data.test_copy.copy()
        preds['Predictions'] = x_pred

        scored = pd.DataFrame(index=preds.index)
        scored['Predictions'] = x_pred

        scored['Threshold'] = 0.5
        if imported_data.type_of_split == "Testing":
            scored['Actual'] = imported_data.y_test
            scored.sort_index(inplace=True)
            scored.plot(figsize=(16, 9), style=['b-', 'r-', 'g--'])
        else:
            scored.sort_index(inplace=True)
            scored.plot(figsize=(16, 9), style=['b-', 'r-'])
        plt.ylim(-1, 1.5)
        plt.show()

        normal = 0
        attack = 0

        true_attack = 0
        true_normal = 0
        false_attack = 0
        false_normal = 0

        preds.reset_index(inplace=True)
        preds["Predictions"] = preds["Predictions"].astype(str)

        if imported_data.type_of_split == "Regular":
            if imported_data.unique_values_normal_attack[0] == "Normal":
                for index, value in enumerate(x_pred[:, 0]):
                    if value <= 0.5:
                        preds.at[index, "Predictions"] = "Normal"
                        normal = normal + 1
                    elif value > 0.5:
                        preds.at[index, "Predictions"] = "Attack"
                        attack = attack + 1
            else:
                for index, value in enumerate(x_pred[:, 0]):
                    if value >= 0.5:
                        preds.at[index, "Predictions"] = "Normal"
                        normal = normal + 1
                    elif value < 0.5:
                        preds.at[index, "Predictions"] = "Attack"
                        attack = attack + 1
            normal_percent = (100 * normal) / (normal + attack)
            print("All: {:d}, normal: {:d} ({:.4f}%), attack: {:d} ({:.4f}%)".format(
                (normal + attack), normal, normal_percent, attack,
                (100 - normal_percent)))

        else:
            if imported_data.unique_values_normal_attack[0] == "Normal":
                for index, value in enumerate(x_pred[:, 0]):
                    if value <= 0.5:
                        preds.at[index, "Predictions"] = "Normal"
                        if preds.at[index, 'Normal/Attack'] == "Normal":
                            true_normal = true_normal + 1
                        else:
                            false_normal = false_normal + 1
                    elif value > 0.5:
                        preds.at[index, "Predictions"] = "Attack"
                        if preds.at[index, 'Normal/Attack'] == "Attack":
                            true_attack = true_attack + 1
                        else:
                            false_attack = false_attack + 1
            else:
                for index, value in enumerate(x_pred[:, 0]):
                    if value >= 0.5:
                        preds.at[index, "Predictions"] = "Normal"
                        if preds.at[index, 'Normal/Attack'] == "Normal":
                            true_normal = true_normal + 1
                        else:
                            false_normal = false_normal + 1
                    elif value < 0.5:
                        preds.at[index, "Predictions"] = "Attack"
                        if preds.at[index, 'Normal/Attack'] == "Attack":
                            true_attack = true_attack + 1
                        else:
                            false_attack = false_attack + 1
            correct = true_normal + true_attack
            incorrect = false_normal + false_attack
            correct_percent = (100 * correct) / (correct + incorrect)

            accuracy = correct / (correct + incorrect)
            sensitivity = true_attack / (false_normal + true_attack)
            precision = true_attack / (true_attack + false_attack)
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
            print('*************** Evaluation on Test Data ***************')
            print("All: {:d}, correct: {:d} ({:.4f}%), incorrect: {:d} ({:.4f}%)\nTesting Accuracy: {:.2f}\nSensitivity(Recall): {:.2f}\nPrecision: {:.2f}\nF1-Score: {:.2f}".format
                ((correct + incorrect), correct, correct_percent, incorrect, (100 - correct_percent), accuracy,
                 sensitivity, precision, f1_score))
            print('--------------------------------------------------------')
            print("")

            with np.nditer(x_pred, op_flags=['readwrite']) as it:
                for val in it:
                    if val != 0 and val != 1:
                        if val < 0.5:
                            val[...] = 0
                        else:
                            val[...] = 1
            print('*************** Confusion matrix ***************')
            print(confusion_matrix(imported_data.y_test, x_pred))
            print('--------------------------------------------------------')
            print("")
            
        preds.to_csv("Predictions{:s}.csv".format(self.model_name))
