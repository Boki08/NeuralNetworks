import keras.models
import keras.layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import initializers
from keras_self_attention import SeqSelfAttention
from sklearn.metrics import classification_report, confusion_matrix


class Model:
    def __init__(self):
        self.kerasModel = keras.models.Sequential()
        self.model_name = "model"
        self.models = {}

    def createClassicLSTMModel(self, data):
        ######################################################## classic
        self.kerasModel = keras.models.Sequential()
        lstm_out = 196
        self.kerasModel.add(keras.layers.LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.1))
        self.kerasModel.add(keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = self.kerasModel.fit(data.x_train, data.y_train, batch_size=128, epochs=20, validation_split=0.1, verbose=1)

        print(self.kerasModel.summary())
        train_val_loss(history)
        train_val_accuracy(history)

        self.model_name = "ClassicLSTMModel"
        self.models[self.model_name] = self.kerasModel
        if data.type_of_split == "Training":
            self.evaluation(data)

        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}\n".format(self.model_name))
        ########################################################

    def createStackedLSTMModel(self, data):
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

        history = self.kerasModel.fit(data.x_train, data.y_train, epochs=25, batch_size=128, validation_split=0.1, verbose=1)

        print(self.kerasModel.summary())
        train_val_loss(history)
        train_val_accuracy(history)

        self.model_name = "StackedLSTMModel"
        self.models[self.model_name] = self.kerasModel
        if data.type_of_split == "Training":
            self.evaluation(data)

        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}\n".format(self.model_name))
        ########################################################

    def createBidirectionalLSTMModel(self, data):
        ######################################################## Bidirectional
        self.kerasModel = keras.models.Sequential()
        lstm_out = 64
        self.kerasModel.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_out, return_sequences=True)))
        self.kerasModel.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_out)))
        self.kerasModel.add(keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        history = self.kerasModel.fit(data.x_train, data.y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1)

        print(self.kerasModel.summary())
        train_val_loss(history)
        train_val_accuracy(history)

        self.model_name = "BidirectionalLSTMModel"
        self.models[self.model_name] = self.kerasModel
        if data.type_of_split == "Training":
            self.evaluation(data)

        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}\n".format(self.model_name))
        ########################################################

    def createGRUModel(self, data):
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


        history = self.kerasModel.fit(data.x_train, data.y_train, epochs=20, batch_size=25, validation_split=0.1, verbose=1)

        print(self.kerasModel.summary())
        train_val_loss(history)
        train_val_accuracy(history)

        self.model_name = "GRUModel"
        self.models[self.model_name] = self.kerasModel
        if data.type_of_split == "Training":
            self.evaluation(data)

        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}\n".format(self.model_name))
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

                print("\tLoaded {:s}\n".format(self.model_name))
            except OSError:
                self.model_name = temp_name
                print("Classic Model does not exist")
            except BaseException as err:
                self.model_name = temp_name
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        else:
            print("\tLoaded {:s}\n".format(self.model_name))
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

                print("\tLoaded {:s}\n".format(self.model_name))
            except OSError:
                self.model_name = temp_name
                print("Stacked Model does not exist")
            except BaseException as err:
                self.model_name = temp_name
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        else:
            print("\tLoaded {:s}\n".format(self.model_name))
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

                print("\tLoaded {:s}\n".format(self.model_name))
            except OSError:
                self.model_name = temp_name
                print("Bidirectional Model does not exist")
            except BaseException as err:
                self.model_name = temp_name
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        else:
            print("\tLoaded {:s}\n".format(self.model_name))
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

                print("\tLoaded {:s}\n".format(self.model_name))
            except OSError:
                self.model_name = temp_name
                print("GRU Model does not exist")
            except BaseException as err:
                self.model_name = temp_name
                print(f"Unexpected {err=}, {type(err)=}")
                raise
        else:
            print("\tLoaded {:s}\n".format(self.model_name))
        ########################################################

    def simulate(self, imported_data):

        print("\tUsing {:s}\n".format(self.model_name))

        x_input = np.array(imported_data.x_test)
        x_input = x_input.reshape((x_input.shape[0], imported_data.n_steps, imported_data.n_features))

        x_pred = self.models[self.model_name].predict(x_input)
        x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1])

        preds = imported_data.test_copy.copy()
        preds = preds.assign(Predictions=x_pred, PredictionValues=x_pred)

        scored = pd.DataFrame(index=preds.index)
        scored = scored.assign(Predictions=x_pred, Threshold=0.5)
        scored.sort_index(inplace=True)

        if imported_data.type_of_split == "Training":
            scored['Actual'] = imported_data.y_test
            scored.plot(figsize=(16, 9), style=['b-', 'r-', 'g--'])
        else:
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
            print('--------------------------------------------------------\n')

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
            print('--------------------------------------------------------\n')

            with np.nditer(x_pred, op_flags=['readwrite']) as it:
                for val in it:
                    if val != 0 and val != 1:
                        if val < 0.5:
                            val[...] = 0
                        else:
                            val[...] = 1
            print('*************** Confusion matrix ***************')
            print(confusion_matrix(imported_data.y_test, x_pred))
            print('--------------------------------------------------------\n')

        print('Writing to a file...')
        preds.to_csv("Predictions{:s}_{:s}.csv".format(self.model_name, imported_data.type_of_split), float_format="%.8f")
        print('Done\n')
        
    def evaluation(self, data):
        x_input = np.array(data.x_train)
        x_input = x_input.reshape((x_input.shape[0], data.n_steps, data.n_features))
        pred_labels_tr = self.models[self.model_name].predict(x_input)
        pred_labels_tr = pred_labels_tr.reshape(pred_labels_tr.shape[0], pred_labels_tr.shape[1])

        with np.nditer(pred_labels_tr, op_flags=['readwrite']) as it:
            for val in it:
                if val != 0 and val != 1:
                    if val < 0.5:
                        val[...] = 0
                    else:
                        val[...] = 1

        x_input = np.array(data.x_test)
        x_input = x_input.reshape((x_input.shape[0], data.n_steps, data.n_features))
        pred_labels_te = self.models[self.model_name].predict(x_input)
        pred_labels_te = pred_labels_te.reshape(pred_labels_te.shape[0], pred_labels_te.shape[1])

        with np.nditer(pred_labels_te, op_flags=['readwrite']) as it:
            for val in it:
                if val != 0 and val != 1:
                    if val < 0.5:
                        val[...] = 0
                    else:
                        val[...] = 1

        print('*************** Evaluation on Test Data ***************')
        print(classification_report(data.y_test, pred_labels_te))
        print('--------------------------------------------------------\n')

        print('*************** Evaluation on Training Data ***************')
        print(classification_report(data.y_train, pred_labels_tr))
        print('--------------------------------------------------------\n')

        print('*************** Confusion matrix ***************')
        print(confusion_matrix(data.y_test, pred_labels_te))
        print('--------------------------------------------------------\n')

def train_val_loss(history):
    loss_train = history.history['loss']
    loss_val = history.history['val_loss']
    plt.plot(loss_train, 'y', label='Training loss')
    plt.plot(loss_val, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_val_accuracy(history):
    loss_train = history.history['accuracy']
    loss_val = history.history['val_accuracy']
    plt.plot(loss_train, 'g', label='Training accuracy')
    plt.plot(loss_val, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
