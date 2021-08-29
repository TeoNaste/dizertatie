from controllers.ResultsController import ResultsController
from controllers.TrainerController import TrainerController
from dataLoaders.FileDataLoader import FileDataLoader
from dataLoaders.ResultsLoader import ResultsLoader
from gui.StartMenu import StarMenu
from preprocessing.DataPreprocessor import DataPreprocessor
from utils.SharedData import SharedData


def main():
    data_preprocessor = DataPreprocessor()
    data_loader = FileDataLoader(data_preprocessor)
    results_loader = ResultsLoader()
    shared_data = SharedData()
    results_controller = ResultsController(results_loader)
    controller = TrainerController(data_loader,shared_data,results_controller)
    starMenu = StarMenu(controller,results_controller)
    starMenu.start()


if __name__ == '__main__':
    main()
