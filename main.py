import pandas as pd
import numpy as np
import seaborn as sns
import sys
import model
import data

sns.set(color_codes=True)

if __name__ == '__main__':
    # fix random seed for reproducibility
    np.random.seed(7)

    trainFile = 'FullTrain.csv'
    testFile = 'test.csv'

    model = model.Model()
    data = data.Data(trainFile, testFile)
    if not data.imported:
        print("One of the Data files does not exist!")
        sys.exit()

    data.importFiles()

    features = pd.read_csv(trainFile, sep=';')
    featuresCopy = features

    test2 = pd.read_csv(testFile, sep=';')
    test2Copy = test2

    menu = {'1': "Import Data", '2': "Create Model", '3': "Load Model", '4': "Simulate", '5': "Exit"}
    modelMenu = {'1': "Classic LSTM model", '2': "Stacked LSTM model", '3': "Bidirectional LSTM model", '4': "GRU model",
                 '5': "Exit"}
    dataMenu = {'1': "Normal", '2': "Test"}
    while True:
        options = menu.keys()
        for entry in options:
            print(entry, menu[entry])

        selection = input("Please Select:")
        if selection == '1':
            while True:
                options = dataMenu.keys()
                for entry in options:
                    print(entry, dataMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    data.split_data("Regular")
                    break
                elif selection == '2':
                    data.split_data("Testing")
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '2':
            while True:
                options = modelMenu.keys()
                for entry in options:
                    print(entry, modelMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    model.createClassicLSTMModel(data.x_train, data.y_train)
                    break
                elif selection == '2':
                    model.createStackedLSTMModel(data.x_train, data.y_train)
                    break
                elif selection == '3':
                    model.createBidirectionalLSTMModel(data.x_train, data.y_train)
                    break
                elif selection == '4':
                    model.createGRUModel(data.x_train, data.y_train)
                    break
                elif selection == '5':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '3':
            while True:
                options = modelMenu.keys()
                for entry in options:
                    print(entry, modelMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    model.loadClassicLSTMModel()
                    break
                elif selection == '2':
                    model.loadStackedLSTMModel()
                    break
                elif selection == '3':
                    model.loadBidirectionalLSTMModel()
                    break
                elif selection == '4':
                    model.loadGRUModel()
                    break
                elif selection == '5':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '4':
            if model.kerasModel.built:
                model.simulate(data)
            else:
                print("Model is not loaded!")
        elif selection == '5':
            break
        else:
            print("Unknown Option Selected!")
