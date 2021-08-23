import unittest
from unittest import mock
from unittest.mock import MagicMock

from controllers.TrainerController import TrainerController
import numpy as np

from models.ModelSimilarity import ModelSimilarity
from utils.SharedData import SharedData


class ControllerTests(unittest.TestCase):

    @mock.patch('dataLoaders.FileDataLoader', autospec=True)
    @mock.patch('utils.SharedData', autospec=True)
    def test_most_freq_words(self,mock_data_loader,mock_shared_data):
        dataset = np.array([
            [0,'coronavirus group represent group english', 'expls'],
            [1,'man died coronavirus vaccine','exp'],
            [2,'coronavirus vaccine man man coronavirus','expl'],
            [3,'man died coronavirus vaccine actually heart attack unrelated vaccine','expl']])
        n = 5
        vocab = ['coronavirus', 'man', 'died', 'vaccine', 'pandemic']
        mock_shared_data.vocab = MagicMock(return_value=vocab)
        controller = TrainerController(mock_data_loader,mock_shared_data)

        result = controller.compute_most_frequent_words_vocabulary(dataset,n)

        self.assertEqual(len(result),n)
        self.assertIn('coronavirus',result)
        self.assertIn('man',result)
        self.assertIn('died',result)

    @mock.patch('dataLoaders.FileDataLoader', autospec=True)
    @mock.patch('utils.SharedData', autospec=True)
    def test_BoW_use_tf(self, mock_data_loader,mock_shared_data):
        dataset = np.array([
            [0, 'coronavirus group represent group english', 'expls'],
            [1, 'man died coronavirus vaccine', 'exp'],
            [2, 'coronavirus vaccine man man coronavirus', 'expl'],
            [3, 'man died coronavirus vaccine actually heart attack unrelated vaccine', 'expl']])
        vocab = ['coronavirus', 'man', 'died', 'vaccine', 'pandemic']
        mock_shared_data.vocab = MagicMock(return_value=vocab)
        controller = TrainerController(mock_data_loader,mock_shared_data)

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

        vocab = ['coronavirus', 'man', 'died', 'vaccine', 'pandemic']
        shared_data = SharedData()
        shared_data.vocab = vocab
        mock_data_loader.load_dataset = MagicMock(return_value=(knowledge, [0,0,1]))

        model_similarity = ModelSimilarity(mock_data_loader).create_model(2)
        controller = TrainerController(mock_data_loader, shared_data)
        controller.vocab = controller.compute_most_frequent_words_vocabulary(dataset,5)

        result = controller.create_dataset_for_similarity_models(dataset,labels,model_similarity)

        self.assertEqual(result.shape[1],12)
        self.assertEqual(result.shape[0],len(dataset))

    @mock.patch('dataLoaders.FileDataLoader', autospec=True)
    @mock.patch('utils.SharedData', autospec=True)
    def test_to_word_embeddings(self, mock_data_loader,mock_shared_data):
        dataset = np.array([
            [0, 'coronavirus group represent group english', 'expls'],
            [1, 'man died coronavirus vaccine', 'exp'],
            [2, 'coronavirus vaccine man man coronavirus', 'expl'],
            [3, 'man died coronavirus vaccine actually heart attack unrelated vaccine', 'expl']])
        vocab = ['coronavirus', 'man', 'died', 'vaccine', 'pandemic']
        mock_shared_data.vocab = MagicMock(return_value=vocab)
        controller = TrainerController(mock_data_loader,mock_shared_data)
        embedding_indexes = {'coronavirus' : [0.123,0.323,0.231,0.5], 'man': [0.456,0.323,0.5,0.78], 'died':[0.3,0.1,0.89,0.44], 'vaccine':[0.78,0.45,0.23,0.23], 'pandemic': [0.56,0.78,0.12,0.123]}
        mock_data_loader.load_embedding_indexes = MagicMock(return_value=(embedding_indexes))
        embedding_dim = 3

        result_index, result_matrix = controller.text_to_word_embeddings(vocab,embedding_dim,dataset)

        self.assertEqual(result_index.shape[0],embedding_dim+1)

    @mock.patch('dataLoaders.FileDataLoader', autospec=True)
    @mock.patch('utils.SharedData',autospec=True)
    def test_text_to_indexes(self, mock_data_loader,mock_shared_data):
        dataset = np.array([
            [0, 'coronavirus group represent group english', 'expls'],
            [1, 'man died coronavirus vaccine', 'exp'],
            [2, 'coronavirus vaccine man man coronavirus', 'expl'],
            [3, 'man died coronavirus vaccine actually heart attack unrelated vaccine', 'expl']])
        vocab = ['coronavirus', 'man', 'died', 'vaccine', 'pandemic']
        mock_shared_data.vocab = MagicMock(return_value=vocab)
        controller = TrainerController(mock_data_loader,mock_shared_data)
        word_count = 11
        text_data = [item[1] for item in dataset]

        result_indexes, result_weights = controller.text_to_indexes(len(vocab),text_data)

        self.assertEqual(len(result_weights),word_count)
        self.assertEqual(len(result_indexes),len(dataset))
