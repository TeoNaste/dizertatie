import unittest
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
from models.ModelSimilarity import ModelSimilarity


class ModelSimilarityTests(unittest.TestCase):

    @mock.patch('dataLoaders.FileDataLoader',autospec=True)
    def test_cosine_similarity(self,mock_data_loader):
        documents= np.array([
            'group represent colombian negotiation pfizer speak english',
            'man died coronavirus vaccine',
            'coronavirus vaccine nullifies 2.79% population miinterpretation u.s. cdc report',
            'man died coronavirus vaccine actually heart attack unrelated vaccine'
        ])

        labels = [0,0,1,1]
        mock_data_loader.load_dataset = MagicMock(return_value=(documents,labels))
        model_similarity = ModelSimilarity(mock_data_loader)

        result = model_similarity.compute_cosine_similarity(documents)

        self.assertEqual(result.shape,(4,4))

    @mock.patch('dataLoaders.FileDataLoader',autospec=True)
    def test_get_top_similar(self,mock_data_loader):
        documents = np.array([
            'group represent colombian negotiation pfizer speak english',
            'man died coronavirus vaccine',
            'coronavirus vaccine nullifies 2.79% population miinterpretation u.s. cdc report',
        ])
        sentence = 'man died coronavirus vaccine actually heart attack unrelated vaccine'
        label = 1

        labels = [0, 0, 1]
        mock_data_loader.load_dataset = MagicMock(return_value=(documents, labels))
        model_similarity = ModelSimilarity(mock_data_loader)

        claims,labels = model_similarity.get_top_similar(sentence,label)

        self.assertEqual(len(claims),3)
        self.assertEqual(claims[0],documents[1])
        self.assertEqual(labels[0],0)


if __name__ == '__main__':
    unittest.main()