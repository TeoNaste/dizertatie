


class ResultsLoader:

    def __init__(self):
        self.filename = 'savedModels/best_results.txt'

    def load_best_results(self):
        best_models = []
        with open(self.filename,'r') as file:
            for line in file.readlines():
                model = line.split(',')[0]
                accuracy = float(line.split(',')[1])
                loss = float(line.split(',')[2])
                date = line.split(',')[3]

                from controllers.ResultsController import ModelInfo
                best_models.append(ModelInfo(model,accuracy,loss,date))

        return best_models

    def write_best_results(self,best_models):
        f = open(self.filename,"a")
        for model in best_models:
            f.write(str(model))
        f.close()
