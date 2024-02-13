""" This is the main script that is used to run the language detector. It takes the path to the csv file, the path to save the model, the path to save the cross validation and a boolean to indicate if we want to retrain the model or not. """
import joblib
import data_reader_analyser_class as drac
import sys
import model_trainer as mt


if __name__ == "__main__":
    # get the paths from the command line
    path_to_csv = sys.argv[1]
    path_to_save_model = sys.argv[2]
    path_to_save_cv = sys.argv[3]
    train_model = bool(int(sys.argv[4]))

    # read and analyse the data
    data_reader_analyser = drac.DataReaderAnalyser(path_to_csv)
    data_reader_analyser.run()
    dataset = data_reader_analyser.get_data_set()

    # in case we want to retrain the model
    if train_model:
        # train the model
        model_trainer = mt.ModelTrainer(dataset, path_to_save_model, path_to_save_cv)
        model_trainer.train_model()

    # load the best model and cv
    loaded_model = joblib.load(path_to_save_model)
    loaded_cv = joblib.load(path_to_save_cv)

    # predict the language of the input text
    print("===========Language Detector============")
    user = input("Enter a Text: ")
    data = loaded_cv.transform([user])
    output = loaded_model.predict(data)
    print("The detected language of the input text is : " + output[0])
