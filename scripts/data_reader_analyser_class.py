""" This module contains the DataReaderAnalyser class which is used to read and analyse the dataset """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataReaderAnalyser:
    """A class to read and analyse the dataset.

    In this class, we read the dataset from a csv file and analyse it. We check for the total rows in the dataset, if there is any 'Language' or 'Text' row with Nan data, the total different languages, the unique languages, the count of rows of a specific language and generate plots for the language counts.

    Attributes:
        data_set (pd.DataFrame): The dataset to be read and analysed

    """

    def __init__(self, file_path: str) -> None:
        """The constructor for the DataReaderAnalyser class.
        Args:
            file_path (str): The path to the csv file
        """
        self.data_set = pd.read_csv(file_path)

    def total_rows_in_dataset(self) -> int:
        """A method to get the total rows in the dataset.
        Returns:
            int: The total rows in the dataset
        """
        return self.data_set.size

    def check_row_nan(self, column_name: str) -> pd.DataFrame:
        """A method to check if there is any Nan data in a specific column.
        Args:
            column_name (str): The name of the column to be checked
        Returns:
            pd.DataFrame: The rows with Nan data
        """
        return self.data_set[self.data_set[column_name].isnull()]

    def total_different_languages(self) -> int:
        """A method to get the total different languages in the dataset.
        Returns:
            int: The total different languages in the dataset
        """
        return self.data_set["Language"].nunique()

    def unique_languages(self) -> np.ndarray:
        """A method to get the unique languages in the dataset.
        Returns:
            np.ndarray: The unique languages in the dataset
        """
        return self.data_set["Language"].unique()

    def count_rows_of_language(self, language: str) -> int:
        """A method to get the count of rows of a specific language.
        Args:
            language (str): The language to be checked
        Returns:
            int: The count of rows of the specific language
        """
        return self.data_set[self.data_set["Language"] == language]["Language"].size

    def generate_plots_for_language_counts(self, languages: list) -> None:
        """A method to generate plots for the language counts.
        Args:
            languages (list): The list of languages to be plotted
        """
        for language in languages:
            plt.bar(language, self.count_rows_of_language(language))
        plt.savefig("language_count.png")

    def print_escape_line(self) -> None:
        """A method to print an escape line."""
        print("=================================")

    def get_data_set(self) -> pd.DataFrame:
        """A method to get the dataset.
        Returns:
            pd.DataFrame: The dataset
        """
        return self.data_set

    def run(self) -> None:
        """A method to run the DataReaderAnalyser class and print the results."""
        print("Data Set Overview")
        print(self.data_set.head())
        self.print_escape_line()

        print("Total rows in the Dataset = " + str(self.total_rows_in_dataset()))
        self.print_escape_line()

        print(
            "If there is any 'Language' row with Nan data: "
            + str(self.check_row_nan("Language"))
        )
        print(
            "If there is any 'Text' row with Nan data: "
            + str(self.check_row_nan("Text"))
        )
        self.print_escape_line()

        print(str(self.total_different_languages()) + " total different languages")
        print(self.unique_languages())
        self.print_escape_line()

        print("English = " + str(self.count_rows_of_language("English")))
        print("German = " + str(self.count_rows_of_language("German")))
        self.print_escape_line()

        print("Generating plots for language counts under 'language_count.png'")
        self.generate_plots_for_language_counts(
            ["English", "German", "French", "Spanish", "Italian", "Portugeese", "Dutch"]
        )
        print("Plots generated successfully!")
        self.print_escape_line()
