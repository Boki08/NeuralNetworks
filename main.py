import pandas as pd
import numpy as np
import sys
import model
import data


if __name__ == '__main__':
    # fix random seed for reproducibility
    np.random.seed(7)

    trainFile = 'FullTrain.csv'
    #testFile = '20191206MixTest.csv'
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

    menu = {'1.': "Import Data", '2.': "Create Model", '3.': "Load Model", '4.': "Simulate", '5.': "Exit"}
    modelMenu = {'1.': "Classic LSTM model", '2.': "Stacked LSTM model", '3.': "Bidirectional LSTM model", '4.': "GRU model",
                 '5.': "Exit"}
    dataMenu = {'1.': "Regular", '2.': "Training", '3.': "Exit"}
    while True:
        print('************************************************')
        print('Menu:')
        options = menu.keys()
        for entry in options:
            print(entry, menu[entry])

        selection = input("Please Select:")
        if selection == '1':
            print('--------------------------------------------------------')
            print("Import Data")
            while True:
                options = dataMenu.keys()
                for entry in options:
                    print(entry, dataMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    data.split_data("Regular")
                    break
                elif selection == '2':
                    data.split_data("Training")
                    break
                elif selection == '3':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '2':
            print('--------------------------------------------------------')
            print("Create Model")
            while True:
                options = modelMenu.keys()
                for entry in options:
                    print(entry, modelMenu[entry])

                selection = input("Please Select:")
                if selection == '1':
                    model.createClassicLSTMModel(data)
                    break
                elif selection == '2':
                    model.createStackedLSTMModel(data)
                    break
                elif selection == '3':
                    model.createBidirectionalLSTMModel(data)
                    break
                elif selection == '4':
                    model.createGRUModel(data)
                    break
                elif selection == '5':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '3':
            print('--------------------------------------------------------')
            print("Load Model")
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
            print('--------------------------------------------------------')
            print("Simulate")
            if model.kerasModel.built:
                model.simulate(data)
            else:
                print("\tModel is not loaded!")
        elif selection == '5':
            break
        else:
            print("Unknown Option Selected!")
