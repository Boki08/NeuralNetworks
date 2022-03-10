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

    model_obj = model.Model()
    data_obj = data.Data(trainFile, testFile)
    if not data_obj.imported_train:
        print("Train file does not exist!")
        sys.exit()
    if not data_obj.imported_test:
        print("Test file does not exist!")

    data_obj.split_data(data_obj.type_of_split)


    menu = {'1.': "Import Data", '2.': "Create Model", '3.': "Load Model", '4.': "Simulate", '5.': "Exit"}
    modelMenu = {'1.': "Classic LSTM model", '2.': "Stacked LSTM model", '3.': "Bidirectional LSTM model", '4.': "GRU model",
                 '5.': "Exit"}
    dataMenu = { '1.': "Train", '2.': "Regular", '3.': "Exit"}
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
                    data_obj.split_data("Train")
                    break
                elif selection == '2':
                    if data_obj.imported_test:
                        data_obj.split_data("Regular")
                        break
                    else:
                        print("Test file does not exist!")
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
                    model_obj.create_classic_lstm_model(data_obj)
                    break
                elif selection == '2':
                    model_obj.create_stacked_lstm_model(data_obj)
                    break
                elif selection == '3':
                    model_obj.create_bidirectional_lstm_model(data_obj)
                    break
                elif selection == '4':
                    model_obj.create_gru_model(data_obj)
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
                    model_obj.load_classic_lstm_model()
                    break
                elif selection == '2':
                    model_obj.load_stacked_lstm_model()
                    break
                elif selection == '3':
                    model_obj.load_bidirectional_lstm_model()
                    break
                elif selection == '4':
                    model_obj.load_gru_model()
                    break
                elif selection == '5':
                    break
                else:
                    print("Unknown Option Selected!")
        elif selection == '4':
            print('--------------------------------------------------------')
            print("Simulate")
            if model_obj.kerasModel.built:
                if data_obj.type_of_split == "Train":
                    print("Simulate on Validation Data")
                    model_obj.simulate_val(data_obj)
                else:
                    print("Simulate on Test Data")
                    model_obj.simulate(data_obj)
            else:
                print("\tModel is not loaded!")
        elif selection == '5':
            break
        else:
            print("Unknown Option Selected!")
