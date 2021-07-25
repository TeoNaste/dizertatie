import unittest
from unittest import mock
from unittest.mock import MagicMock

from controllers.Controller import Controller
import numpy as np

from models.ModelSimilarity import ModelSimilarity


class ControllerTests(unittest.TestCase):

    @mock.patch('dataLoaders.FileDataLoader', autospec=True)
    def test_most_freq_words(self,mock_data_loader):
        dataset = np.array([
            [0,'coronavirus group represent group english', 'expls'],
            [1,'man died coronavirus vaccine','exp'],
            [2,'coronavirus vaccine man man coronavirus','expl'],
            [3,'man died coronavirus vaccine actually heart attack unrelated vaccine','expl']])
        n = 5
        controller = Controller(mock_data_loader)

        result = controller.compute_most_frequent_words_vocabulary(dataset,n)

        self.assertEqual(len(result),n)
        self.assertIn('coronavirus',result)
        self.assertIn('man',result)
        self.assertIn('died',result)

    @mock.patch('dataLoaders.FileDataLoader', autospec=True)
    def test_BoW_use_tf(self, mock_data_loader):
        dataset = np.array([
            [0, 'coronavirus group represent group english', 'expls'],
            [1, 'man died coronavirus vaccine', 'exp'],
            [2, 'coronavirus vaccine man man coronavirus', 'expl'],
            [3, 'man died coronavirus vaccine actually heart attack unrelated vaccine', 'expl']])
        controller = Controller(mock_data_loader)
        vocab = ['coronavirus','man','died','vaccine','pandemic']

        result = controller.text_to_bag_of_words(vocab,dataset)

        self.assertEqual(len(dataset),len(result))

    @mock.patch('dataLoaders.FileDataLoader', autospec=True)
    def test_create_data_for_similarity(self, mock_data_loader):
        knowledge = np.array([
            [0, 'coronavirus group represent group english', 'expls'],
            [1, 'man died coronavirus vaccine', 'exp'],
            [2, 'coronavirus vaccine man man coronavirus', 'expl']])

        dataset = np.array([[3, 'man died coronavirus vaccine actually heart attack unrelated vaccine', 'expl'],
                            [4, 'coronavirus pandemic global panic', 'expl']])
        labels = np.array([1,1])
        mock_data_loader.load_dataset = MagicMock(return_value=(knowledge, [0,0,1]))

        model_similarity = ModelSimilarity(mock_data_loader).create_model(2)
        controller = Controller(mock_data_loader)
        controller.vocab = controller.compute_most_frequent_words_vocabulary(dataset,5)

        result = controller.create_dataset_for_similarity_models(dataset,labels,model_similarity)

        self.assertEqual(result.shape[1],12)
        self.assertEqual(result.shape[0],len(dataset))

