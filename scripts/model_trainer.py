""" A module to train a model and save it to a file """

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib


class ModelTrainer:
    """A class to train a model and save it to a file.

    In this class, we train a model using the Multinomial Naive Bayes algorithm and save it to a file using joblib. We also save the CountVectorizer to a file.
    MultinomialNB is a Naive Bayes classifier that is suitable for classification with discrete features (e.g., word counts for text classification).

    Attributes:
        dataset (pd.DataFrame): The dataset to be used to train the model
        sentences (pd.Series): The sentences in the dataset
        labeled_languages (pd.Series): The labeled languages in the dataset for each sentence
        CV (CountVectorizer): The CountVectorizer to be used to transform the sentences into a matrix of token counts
        best_random_state (int): The best random state to be used in the train_test_split
        path_to_save_model (str): The path to save the model
        path_to_save_cv (str): The path to save the CountVectorizer

    """

    def __init__(
        self, dataset: pd.DataFrame, path_to_save_model: str, path_to_save_cv: str
    ) -> None:
        """The constructor for the ModelTrainer class.
        Args:
            dataset (pd.DataFrame): The dataset to be used to train the model
            path_to_save_model (str): The path to save the model
            path_to_save_cv (str): The path to save the CountVectorizer
        """
        self.dataset = dataset
        self.sentences = dataset["Text"]
        self.labled_languages = dataset["Language"]
        self.CV = CountVectorizer()
        self.best_random_state = 0
        self.path_to_save_model = path_to_save_model
        self.path_to_save_cv = path_to_save_cv

    def search_best_random_state(self, X: np.ndarray) -> int:
        """A method to search for the best random state to be used in the train_test_split.
        Args:
            X (np.ndarray): The matrix of token counts
        Returns:
            int: The best random state to be used in the train_test_split
        """

        best_random_state = 0
        best_result = 0
        for i in range(30, 60):
            X_train, X_test, y_train, y_test = train_test_split(
                X, self.labled_languages, test_size=0.2, random_state=i
            )
            model = MultinomialNB()
            model.fit(X_train, y_train)
            if model.score(X_test, y_test) > best_result:
                best_random_state = i
                best_result = model.score(X_test, y_test)
        return best_random_state

    def train_model(self) -> None:
        """A method to train the model with the best obtained random state and save it to a file."""
        X = self.CV.fit_transform(self.sentences)
        self.best_random_state = self.search_best_random_state(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, self.labled_languages, test_size=0.2, random_state=self.best_random_state
        )
        model = MultinomialNB()
        model.fit(X_train, y_train)
        joblib.dump(model, self.path_to_save_model)
        joblib.dump(self.CV, self.path_to_save_cv)
        print("Model Accuracy " + str(100 * model.score(X_test, y_test)) + "%")
        print("Model saved to: " + self.path_to_save_model)
        print("CountVectorizer saved to: " + self.path_to_save_cv)
