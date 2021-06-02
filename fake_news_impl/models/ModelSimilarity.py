import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing.DataPreprocessor import DataPreprocessor


class ModelSimilarity:

    def __init__(self, preprocessor : DataPreprocessor):
        self.dataPreprocessor = preprocessor

    def predict(self,dataset, df_ir, sentence):
        true_positive = 0
        total = dataset.shape[0]
        # step 1: prepare df_ir - vectorize all documents
        rows = [row['text'] for index, row in df_ir.iterrows()]
        tags = [row['tag'] for index, row in df_ir.iterrows()]
        docIds = [row['docId'] for index, row in df_ir.iterrows()]

        knowledge_base = dict(zip(docIds, tags))

        for row in dataset.iterrows():
            prediction = self.get_most_similar(row, rows, tags)
            tags.append(prediction)

            if prediction == row[1].tag:
                true_positive += 1

        print("IR only acc: ", (true_positive / total) * 100, "%")

        dataset = dataset.append([{'docId': len(docIds) + 1, 'text': sentence, 'tag': ''}])
        prediction = self.get_most_similar(dataset.loc[dataset.shape[0] - 1], rows, tags)
        return prediction

    def get_most_similar(self, current, rows, knowledge_base):
        rows.append(current[1].text)
        vect, vectors = self.dataPreprocessor.vectorize_text(rows, False)
        df_all = pd.DataFrame(vectors.todense(), columns=vect.get_feature_names())

        sim_result = cosine_similarity(df_all)
        df_sim = pd.DataFrame(sim_result)
        last_row = df_sim.tail(1)
        similar_Id = last_row.T.apply(lambda x: x.nlargest(2).idxmin())

        return knowledge_base[similar_Id.iloc[0]]