import torch
from torch import nn
from torchvision import models
from torchvision.models.resnet import ResNet, Bottleneck
import torch.functional as F
import numpy as np
from utils import load_embeddings

class CNNSentenceClassification(nn.Module):
    """
    Architecture inspired by the original paper on convolutional networks for sentence classification (Kim, 2014).
    Architecture and embedding implementation details inspired by Tran in https://chriskhanhtran.github.io/posts/cnn-sentence-classification/.
    """
    def __init__(self, embedding_dim, filter_sizes, n_filters, n_classes, dropout_p, freeze_embedding, embeddings_path='words2index.csv'):
        super().__init__()
        self.embeddings = load_embeddings(embeddings_path, embedding_dim) # Pretrained word embeddings
        self.vocab_size, self.embeddings_dim = self.embeddings.shape # Dimension of embeddings and size of vocabulary

        self.embedding = nn.Embedding.from_pretrained(self.embeddings, freeze=freeze_embedding)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,out_channels=self.n_filters[i],kernel_size=filter_sizes[i]) for i in range(len(filter_sizes))])
        self.fc = nn.Linear(np.sum(n_filters), n_classes)
        self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, x):
        x = self.embedding(x).permute(0,2,1)
        x = [conv(x) for conv in self.convs]
        x = [F.relu(conv) for conv in x]
        x = [F.max_pool1d(conv, kernel_size=conv.shape[2]) for conv in x]
        x = torch.cat([pool.squeeze(dim=2) for pool in x], dim=1)
        x = self.fc(self.dropout(x))

        return x

class RNNSentenceClassification(nn.Module):
    """
    One-directed LSTM for sentence classification.
    Implementation details inspired by Cheng in https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0.
    """
    def __init__(self, embedding_dim, hidden_dim, n_classes, dropout_p, freeze_embedding, embeddings_path='words2index.csv'):
        super().__init__()
        self.embedding_dim = 300
        self.embeddings = load_embeddings(embeddings_path, embedding_dim) # Pretrained word embeddings
        self.vocab_size, self.embeddings_dim = self.embeddings.shape # Dimension of embeddings and size of vocabulary
        self.n_classes = 6 # Number of classes for output
        self.dropout_p = 0.2 # Dropout probability
        self.hidden_dim = 128 # Dimension of hidden states

        # Initialize embedding layer
        self.embedding_layer = nn.Embedding.from_pretrained(self.embeddings, freeze=freeze_embedding)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, n_classes)
        self.dropout = nn.Dropout(p=dropout_p)


    def forward(self, x):
        sent_lens = x[1] # Get sentence lengths
        x = x[0]

        x = self.embedding_layer(x)
        x = nn.utils.rnn.pack_padded_sequence(x, sent_lens, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, sent_lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_forward = x[torch.arange(len(x)),sent_lens - 1,:]
        x_backwards = x[:,0,self.hidden_dim:]
        x = torch.cat((x_forward,x_backwards),1)
        x_fully_connected = self.fc(self.dropout(x_forward))

        return x_fully_connected

class SiameseResnet50(ResNet):
    def __init__(self, denseLayer=None, pretrained=True, dropout_p=0.5):
        super(SiameseResnet50, self).__init__(Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(models.ResNet50_Weights.IMAGENET1K_V2.get_state_dict(True))
        ftrs_in = self.fc.in_features
        self.fc = nn.Linear(ftrs_in, 1584)#nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(ftrs_in, 1584), nn.ReLU(inplace=True))

        if denseLayer:
            self.fc = denseLayer

    def forward_twin(self, x1, x2):
        x1 = self.forward(x1)
        x2 = self.forward(x2)
        return x1, x2
