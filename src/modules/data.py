import torch
import pandas as pd
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import re
import nltk
import json
import csv

class ImageClassificationDataset(Dataset):
    def __init__(self, annotations_file='img_info.csv', img_dir='train', validation=False, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_labels = self.img_labels[self.img_labels['val'] == validation]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = None
        with open(img_path, 'rb') as f:
            image  = Image.open(f).convert('RGB')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

class SentenceClassificationDataset(Dataset):
    """
    Dataset class used for training and evaluating models for sentence classification.
    First, sentences are tokenized and transformed into sequences of indices.
    If a word2index file is provided, it is used to determine the indices of the words.
    """
    def __init__(self, validation=False):

        sentences = []
        class_labels = []
        path = 'valfile.jsonl' if validation else 'trainfile.jsonl'

        with open(path, 'r', encoding="utf-8") as file:
            json_list = list(file)
            for line in json_list:
                json_obj = json.loads(line)
                sentences.append(json_obj['text'])
                class_labels.append(json_obj['label'])

        # Tokenize sentences
        tokenized_sentences = []
        for sentence in sentences:
            sentence = re.sub(r'<.*?>','',sentence)
            sent_tokens = nltk.word_tokenize(sentence)
            if len(sent_tokens) != 0:
                tokenized_sentences.append(sent_tokens)

        # Create vocabulary if no word2index file is available
        word2index = {}
        if os.path.exists('words2index.csv'):
            with open('words2index.csv', 'r',  encoding="utf-8") as file:
                csv_reader = csv.reader(file)
                word2index = {row[0]:int(row[1]) for row in csv_reader}
        else:
            word2index['<pad>'] = 0
            word2index['<unk>'] = 1
            index_count = 2
            for sentence in tokenized_sentences:
                for token in sentence:
                    if token not in word2index:
                        word2index[token] = index_count
                        index_count += 1
            
            # Save vocabulary as csv file
            with open('words2index.csv', 'w',  encoding="utf-8", newline='') as file:
                csv_writer = csv.writer(file)
                for key in word2index.keys():
                    csv_writer.writerow([key, word2index[key]])

        # Encode sentences using vocabulary
        data_points = []
        for sentence in tokenized_sentences:
            data_points.append([word2index.get(token, word2index['<unk>']) for token in sentence])

        # Define data points and labels
        self.X = data_points
        self.y = class_labels

        # Define number of samples
        self.n_samples = len(data_points)

    def __getitem__(self, index):
        return torch.LongTensor(self.X[index]), torch.tensor(self.y[index])

    def __len__(self):
        return self.n_samples