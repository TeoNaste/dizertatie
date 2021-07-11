import unittest
from unittest import mock
from controllers.Controller import Controller
import numpy as np


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