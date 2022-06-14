from ..modules.models import SiameseResnet50, CNNSentenceClassification, RNNSentenceClassification
from ..modules.loss_functions import ContrastiveLoss
from ..modules.data import ImageClassificationDataset, SentenceClassificationDataset
from ..modules.utils import cnn_collate, rnn_collate
from torchvision.models import resnet50
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score


"""
Each entry in the following dictionary represents the parameters of an experiment.
"""
experiment_configs = {
    'resnet_classification_loss': {'model': SiameseResnet50,
                                   'model_params': {'dropout_p': 0.5, 'pretrained': True},
                                   'dataset': ClassificationLossDataset,
                                   'collate_fn': None,
                                   'loss': CrossEntropyLoss,
                                   'optimizer': Adam,
                                   'optimizer_params': {},
                                   'learning_rate': 5e-4,
                                   'scheduler': None,
                                   'scheduler_params': {},
                                   'transform': ToTensor(),
                                   'batch_size': 64,
                                   'epochs': 20,
                                   'eval_function': accuracy_score
    },
    'CNN_sentence_classification': {'model': CNNSentenceClassification,
                                    'model_params': {'embedding_dim': 300,
                                                    'filter_sizes': [2,3,4],
                                                    'n_filters': [100,100,100],
                                                    'n_classes': 6,
                                                    'dropout_p': 0.5,
                                                    'freeze_embedding': True},
                                    'dataset': SentenceClassificationDataset,
                                    'collate_fn': cnn_collate,
                                    'loss': CrossEntropyLoss,
                                    'optimizer': Adam,
                                    'optimizer_params': {'weight_decay': 1e-4},
                                    'learning_rate': 1e-3,
                                    'scheduler': None,
                                    'scheduler_params': {},
                                    'transform': None,
                                    'batch_size': 32,
                                    'epochs': 25,
                                    'eval_function': accuracy_score
    },
    'RNN_sentence_classification': {'model': RNNSentenceClassification,
                                    'model_params': {'embedding_dim': 300,
                                                    'hidden_dim': 128,
                                                    'n_classes': 6,
                                                    'dropout_p': 0.5,
                                                    'freeze_embedding': True},
                                    'dataset': SentenceClassificationDataset,
                                    'collate_fn': rnn_collate,
                                    'loss': CrossEntropyLoss,
                                    'optimizer': Adam,
                                    'optimizer_params': {'weight_decay': 1e-4},
                                    'learning_rate': 1e-3,
                                    'scheduler': None,
                                    'scheduler_params': {},
                                    'transform': None,
                                    'batch_size': 32,
                                    'epochs': 25,
                                    'eval_function': accuracy_score
    }
}