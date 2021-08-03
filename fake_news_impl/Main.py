from controllers.TrainerController import TrainerController
from dataLoaders.FileDataLoader import FileDataLoader
from gui.StartMenu import StarMenu
from preprocessing.DataPreprocessor import DataPreprocessor


def main():
    dataPreprocessor = DataPreprocessor()
    dataLoader = FileDataLoader(dataPreprocessor)
    controller = TrainerController(dataLoader)
    starMenu = StarMenu(controller)
    starMenu.start()


if __name__ == '__main__':
    main()
