import keras.models
import keras.layers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.transforms as transforms
from tensorflow.keras import initializers
from keras_self_attention import SeqSelfAttention
from sklearn.metrics import classification_report, confusion_matrix
from timeit import default_timer as timer
from datetime import timedelta


class Model:
    def __init__(self):
        self.kerasModel = keras.models.Sequential()
        self.model_name = "model"
        self.models = {}

    def create_classic_lstm_model(self, data):
        self.kerasModel = keras.models.Sequential()
        lstm_out = 196
        self.kerasModel.add(keras.layers.LSTM(lstm_out, dropout=0.1, recurrent_dropout=0.1))
        self.kerasModel.add(keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        start = timer()
        history = self.kerasModel.fit(data.x_train, data.y_train, batch_size=128, epochs=50, validation_split=0.1, verbose=1)
        end = timer()

        self.model_name = "ClassicLSTMModel"
        print('* {:s} *'.format(self.model_name).center(65))
        print("_________________________________________________________________")
        print("Training elapsed time:", format(timedelta(seconds=end - start)))
        print(self.kerasModel.summary())

        train_val_loss(history, self.model_name)
        train_val_accuracy(history, self.model_name)

        self.models[self.model_name] = self.kerasModel
        if data.type_of_split == "Train":
            self.evaluation(data)

        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}\n".format(self.model_name))

    def create_stacked_lstm_model(self, data):
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

        start = timer()
        history = self.kerasModel.fit(data.x_train, data.y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1)
        end = timer()

        self.model_name = "StackedLSTMModel"
        print('* {:s} *'.format(self.model_name).center(65))
        print("_________________________________________________________________")
        print("Training elapsed time:", format(timedelta(seconds=end - start)))
        print(self.kerasModel.summary())

        train_val_loss(history, self.model_name)
        train_val_accuracy(history, self.model_name)

        self.models[self.model_name] = self.kerasModel
        if data.type_of_split == "Train":
            self.evaluation(data)

        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}\n".format(self.model_name))

    def create_bidirectional_lstm_model(self, data):
        self.kerasModel = keras.models.Sequential()
        lstm_out = 64
        self.kerasModel.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_out, return_sequences=True)))
        self.kerasModel.add(keras.layers.Bidirectional(keras.layers.LSTM(lstm_out)))
        self.kerasModel.add(keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        start = timer()
        history = self.kerasModel.fit(data.x_train, data.y_train, epochs=20, batch_size=128, validation_split=0.1, verbose=1)
        end = timer()

        self.model_name = "BidirectionalLSTMModel"
        print('* {:s} *'.format(self.model_name).center(65))
        print("_________________________________________________________________")
        print("Training elapsed time:", format(timedelta(seconds=end - start)))
        print(self.kerasModel.summary())

        train_val_loss(history, self.model_name)
        train_val_accuracy(history, self.model_name)

        self.models[self.model_name] = self.kerasModel
        if data.type_of_split == "Train":
            self.evaluation(data)

        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}\n".format(self.model_name))

    def create_gru_model(self, data):
        self.kerasModel = keras.models.Sequential()
        gru_out = 25
        self.kerasModel.add(keras.layers.GRU(gru_out, input_shape=(1, 6), return_sequences=True))
        self.kerasModel.add(keras.layers.GRU(gru_out, activation='relu', return_sequences=True))
        self.kerasModel.add(keras.layers.Dropout(0.5))
        self.kerasModel.add(SeqSelfAttention())
        self.kerasModel.add(keras.layers.GRU(gru_out, return_sequences=False))
        self.kerasModel.add(keras.layers.Dense(1, activation='sigmoid'))
        self.kerasModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        start = timer()
        history = self.kerasModel.fit(data.x_train, data.y_train, epochs=50, batch_size=20, validation_split=0.1, verbose=1)
        end = timer()

        self.model_name = "GRUModel"
        print('* {:s} *'.format(self.model_name).center(65))
        print("_________________________________________________________________")
        print("Training elapsed time:", format(timedelta(seconds=end - start)))
        print(self.kerasModel.summary())

        train_val_loss(history, self.model_name)
        train_val_accuracy(history, self.model_name)

        self.models[self.model_name] = self.kerasModel
        if data.type_of_split == "Train":
            self.evaluation(data)

        self.kerasModel.save(self.model_name)

        print("\tCreated {:s}\n".format(self.model_name))


    def load_classic_lstm_model(self):
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

    def load_stacked_lstm_model(self):
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

    def load_bidirectional_lstm_model(self):
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

    def load_gru_model(self):
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

        fig, ax = plt.subplots()

        scored.plot(figsize=(16, 9), style=['b-', 'r-'], ax=ax)
        plt.ylim(-1, 1.5)

        trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
        ax.text(0.1, -0.25, "{:s}".format("Attack"), color="red", transform=trans, ha="right", va="center")
        ax.text(0.1, 1.25, "{:s}".format("Normal"), color="green", transform=trans, ha="right", va="center")
        plt.show()
        plt.show()

        normal = 0
        attack = 0

        preds.reset_index(inplace=True)

        preds["Predictions"] = preds["Predictions"].astype(str)

        for index, value in enumerate(x_pred[:, 0]):
            if value >= 0.5:
                preds.at[index, "Predictions"] = "Normal"
                normal += 1
            elif value < 0.5:
                preds.at[index, "Predictions"] = "Attack"
                attack += 1
        normal_percent = (100 * normal) / (normal + attack)
        print("All: {:d}, normal: {:d} ({:.4f}%), attack: {:d} ({:.4f}%)".format(
            (normal + attack), normal, normal_percent, attack,
            (100 - normal_percent)))
        print('--------------------------------------------------------\n')

        print('Writing to a file...')
        preds.to_csv("Predictions{:s}_Regular.csv".format(self.model_name), float_format="%.8f")
        print('Done\n')

    def simulate_val(self, imported_data): # validation split

        print("\tUsing {:s}\n".format(self.model_name))

        x_input = np.array(imported_data.x_val)
        x_input = x_input.reshape((x_input.shape[0], imported_data.n_steps, imported_data.n_features))

        x_pred = self.models[self.model_name].predict(x_input)
        x_pred = x_pred.reshape(x_pred.shape[0], x_pred.shape[1])

        preds = imported_data.val_copy.copy()
        preds = preds.assign(Predictions=x_pred, PredictionValues=x_pred)

        with np.nditer(x_pred, op_flags=['readwrite']) as it:
            for val in it:
                if val != 0 and val != 1:
                    if val < 0.5:
                        val[...] = 0
                    else:
                        val[...] = 1

        cf_matrix = confusion_matrix(imported_data.y_val, x_pred)
        print('*************** Confusion matrix ***************')
        print(cf_matrix)
        print('-------------------------------------------------------------\n')
        print_confusion_matrix(cf_matrix, '{:s} (Validation Data)'.format(self.model_name))

        true_attack = cf_matrix[0, 0]
        true_normal = cf_matrix[1, 1]
        false_attack = cf_matrix[1, 0]
        false_normal = cf_matrix[0, 1]

        preds.reset_index(inplace=True)

        preds["Predictions"] = preds["Predictions"].astype(str)

        for index, value in enumerate(x_pred[:, 0]):
            if value >= 0.5:
                preds.at[index, "Predictions"] = "Normal"
            elif value < 0.5:
                preds.at[index, "Predictions"] = "Attack"

        correct = true_normal + true_attack
        incorrect = false_normal + false_attack
        correct_percent = (100 * correct) / (correct + incorrect)

        accuracy = correct / (correct + incorrect)
        sensitivity = true_attack / (false_normal + true_attack)
        precision = true_attack / (true_attack + false_attack)
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

        print('* {:s} *'.format(self.model_name).center(55))
        print('**************** Evaluation on Validation Data ****************')
        print("All: {:d}, correct: {:d} ({:.4f}%), incorrect: {:d} ({:.4f}%)\nTesting Accuracy: {:.4f}\nSensitivity("
              "Recall): {:.4f}\nPrecision: {:.4f}\nF1-Score: {:.4f}".format
              ((correct + incorrect), correct, correct_percent, incorrect, (100 - correct_percent), accuracy,
               sensitivity, precision, f1_score))
        print('---------------------------------------------------------------\n')


        print('Writing to a file...')
        preds.to_csv("Predictions{:s}_Validation.csv".format(self.model_name), float_format="%.8f")
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

        print('* {:s} *'.format(self.model_name).center(55))
        print('*************** Evaluation on Test Data ****************')
        print(classification_report(data.y_test, pred_labels_te))
        print('--------------------------------------------------------\n')

        print('* {:s} *'.format(self.model_name).center(55))
        print('*************** Evaluation on Train Data ***************')
        print(classification_report(data.y_train, pred_labels_tr))
        print('--------------------------------------------------------\n')

        cf_matrix = confusion_matrix(data.y_test, pred_labels_te)
        print('* {:s} *'.format(self.model_name).center(55))
        print('******************* Confusion matrix *******************')
        print(cf_matrix)
        print('--------------------------------------------------------\n')
        print_confusion_matrix(cf_matrix, '{:s} (Test Data)'.format(self.model_name))


def print_confusion_matrix(cf_matrix, model_name):
    group_names = ["True Attack", "False Normal", "False Attack", "True Normal"]
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title("Confusion Matrix - {:s}\n\n".format(model_name))
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")

    ax.xaxis.set_ticklabels(["Attack", "Normal"])
    ax.yaxis.set_ticklabels(["Attack", "Normal"])

    fig = plt.gcf()
    fig.subplots_adjust(left=0.13, right=0.85, bottom=0.13, top=0.85)
    plt.show()


def train_val_loss(history, model_name):
    loss_train = history.history['loss']
    print('******************* Train loss: ')
    print(*loss_train, sep=", ")
    loss_val = history.history['val_loss']
    print('******************* Validation loss: ')
    print(*loss_val, sep=", ")

    plt.plot(loss_train, 'y', label='Train loss')
    plt.plot(loss_val, 'r', label='Validation loss')
    plt.title('Train and Validation loss - {:s}'.format(model_name))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_val_accuracy(history, model_name):
    accuracy_train = history.history['accuracy']
    print('******************* Train accuracy: ')
    print(*accuracy_train, sep=", ")
    accuracy_val = history.history['val_accuracy']
    print('******************* Validation accuracy: ')
    print(*accuracy_val, sep=", ")

    plt.plot(accuracy_train, 'g', label='Train accuracy')
    plt.plot(accuracy_val, 'b', label='Validation accuracy')
    plt.title('Train and Validation accuracy - {:s}'.format(model_name))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
