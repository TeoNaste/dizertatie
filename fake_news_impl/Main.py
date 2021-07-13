from controllers.Controller import Controller
from dataLoaders.FileDataLoader import FileDataLoader
from gui.StartMenu import StarMenu
from preprocessing.DataPreprocessor import DataPreprocessor


def run():
    dataPreprocessor = DataPreprocessor()
    dataLoader = FileDataLoader(dataPreprocessor)
    controller = Controller(dataLoader)
    starMenu = StarMenu(controller)
    starMenu.start()


if __name__ == '__main__':
    run()