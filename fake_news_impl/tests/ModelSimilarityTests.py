import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
from models.ModelSimilarity import ModelSimilarity


class ModelSimilarityTests(unittest.TestCase):

    @mock.patch('dataLoaders.FileDataLoader',autospec=True)
    def test_cosine_similarity(self,mock_data_loader):
        documents = np.array([
            [0, 'coronavirus group represent group english', 'expls'],
            [1, 'man died coronavirus vaccine', 'exp'],
            [2, 'coronavirus vaccine man man coronavirus', 'expl'],
            [3, 'man died coronavirus vaccine actually heart attack unrelated vaccine', 'expl']])

        labels = [0,0,1,1]
        mock_data_loader.load_dataset = MagicMock(return_value=(documents,labels))
        model_similarity = ModelSimilarity(mock_data_loader).create_model()

        result = model_similarity.compute_cosine_similarity(claim[1] for claim in documents)

        self.assertEqual(result.shape,(4,4))

    @mock.patch('dataLoaders.FileDataLoader',autospec=True)
    def test_get_top_similar(self,mock_data_loader):
        documents = np.array([
            [0, 'coronavirus group represent group english', 'expls'],
            [1, 'man died coronavirus vaccine', 'exp'],
            [2, 'coronavirus vaccine man man coronavirus', 'expl']])

        sentence = np.array([3, 'man died coronavirus vaccine actually heart attack unrelated vaccine', 'expl'])
        label = 1

        labels = [0, 0, 1]
        mock_data_loader.load_dataset = MagicMock(return_value=(documents, labels))
        model_similarity = ModelSimilarity(mock_data_loader).create_model()

        claims,labels,sim_values = model_similarity.get_top_similar(sentence,label)

        self.assertEqual(len(claims),3)
        self.assertEqual(claims[0][1],documents[1][1])
        self.assertEqual(labels[0],0)
        self.assertEqual(len(sim_values),3)


if __name__ == '__main__':
    unittest.main()