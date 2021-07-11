import numpy as np

from controllers.Controller import Controller
from dataLoaders.FileDataLoader import FileDataLoader
from gui.StartMenu import StarMenu
from preprocessing.DataPreprocessor import DataPreprocessor

dataset_path = "csvpoynter.csv"
processed_path = "processed.csv"


def run():
    starMenu = StarMenu()
    starMenu.start()

if __name__ == '__main__':
    run()