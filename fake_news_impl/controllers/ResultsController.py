from datetime import datetime

from dataLoaders.ResultsLoader import ResultsLoader


class ModelInfo:

    def __init__(self, model, accuracy, loss, create_date:str):
        self.model = model
        self.accuracy = accuracy
        self.loss = loss
        self.create_date = create_date
        
    def __str__(self):
        return self.model + ',' + \
               str(self.accuracy) + ',' + \
               str(self.loss) + ',' + \
               self.create_date + '\n'


class ResultsController:

    def __init__(self, results_loader: ResultsLoader):
        self.resutls_loader = results_loader
        self.best_models = self.resutls_loader.load_best_results()
        self.last_ranked = None

    def rank_model(self,model,history):
        self.last_ranked = self.__convert_model_to_model_info(model,history)
        if len(self.best_models) == 0:
            self.best_models.append(model)
        else:
            for i,mod in enumerate(self.best_models):
                if history.history['val_accuracy'][-1] > mod.accuracy:
                    self.best_models.insert(i,self.last_ranked)
                    break

        #only keep the first 5 best models
        self.best_models = self.best_models[5:]
        
        self.resutls_loader.write_best_results(self.best_models)

    def __convert_model_to_model_info(self,model,history):
        return ModelInfo(
            model.name,
            history.history['val_accuracy'][-1],
            history.history['val_loss'][-1],
            datetime.now().strftime("%d-%m-%Y %H%M")
        )

