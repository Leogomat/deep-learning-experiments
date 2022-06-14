import numpy as np
import torch
import wandb
import os
from fastprogress import master_bar, progress_bar
from sklearn.metrics import accuracy_score
import csv
from torchtext.vocab import GloVe
from torch.nn.utils.rnn import pad_sequence

@torch.no_grad()
def test_model(model, data_loader, criterion):

    if model.training:
        model.eval()

    pred = []
    true = []
    for inputs, labels in iter(data_loader):
        if next(model.parameters()).is_cuda:
            inputs = inputs.to('cuda')
        true += labels.tolist()
        pred += torch.argmax(model(inputs), dim=1).tolist()

    model.train()

    return criterion(true, pred)

def train_model(model, training_data_loader, validation_data_loader, loss, optimizer, learning_rate, epochs, scheduler=None,
                device='cpu', use_checkpoint=True, checkpoint_folder='', eval_function=accuracy_score):
    """
    This training function is inspired by the one in the exercise notebook.

    David: https://www.youtube.com/watch?v=G7GH0SeNBMA [wandb official youtube channel]
    """

    # Create experiment name
    backbone_name = model.__class__.__name__
    loss_name = loss.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    scheduler_name = scheduler.__class__.__name__ if scheduler else 'NoScheduler'
    experiment_name = f'{backbone_name}-{loss_name}-{optimizer_name}-{scheduler_name}-{learning_rate}'
    checkpoint_path = os.path.join(checkpoint_folder, f'{experiment_name}_checkpoint.pt')

    wandb.login(key="")

    with wandb.init(project="dll-experiments", entity="dll-art", name=experiment_name) as run: 

        # wandb watches gradients and parameters
        wandb.watch(model, log="all", log_freq=1)

        # Run information
        run.config.learning_rate = learning_rate
        run.config.optimizer = optimizer_name
        run.watch(model)

        # Set model to training mode
        model.train()
        if device == 'cuda':
            model.to(device)

        # Check if checkpoint exists. If it does, load it.
        start_epoch = 0
        if os.path.isdir(checkpoint_folder):
            if os.path.isfile(checkpoint_path) and use_checkpoint:
                checkpoint = torch.load(checkpoint_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
        else:
            os.makedirs(checkpoint_folder)

        mb = master_bar(range(start_epoch, epochs))
        for epoch in mb:

            # Here 'inputs' can refer to a list of single images, pairs, or triples (depending on loss function)
            for inputs, labels in progress_bar(iter(training_data_loader), parent=mb):

                # Move batch to GPU (if it is available)
                if device == 'cuda':
                    inputs, labels = inputs.to(device, torch.channels_last), labels.to(device, torch.channels_last)

                # Set gradients to zero
                optimizer.zero_grad(set_to_none=True)

                # Perform the forward pass
                outputs = model(inputs)

                # Compute batch loss and backpropagate
                batch_loss = loss(outputs, labels)
                batch_loss.backward()

                # Log loss to weights and biases
                run.log({'epoch': epoch, 'loss': batch_loss})
                
                # Perform optimization step
                optimizer.step()

            # Perform scheduler step (if necessary)
            if scheduler:
                lr = scheduler.get_last_lr()[0]
                scheduler.step()
            else:
                lr = learning_rate

            # Save checkpoint after each epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)

            # Log current epoch learning rate to weights and biases
            run.log({'learning_rate': lr})

            # Evaluate model according to some criterion
            run.log({'accuracy': test_model(model, validation_data_loader, eval_function)})

def load_embeddings(word2index_path='words2index_cnn.csv', dim=300):
    """
    Load GloVe embeddings as a tensor. If a words2index file is given, it determines the index of the embedding of each word in the resulting
    tensor.
    """

    # Load GloVe embeddings
    glove = GloVe(name='6B', dim=dim)

    # Load word2index dictionary
    word2index = {}
    with open(word2index_path, 'r',  encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        word2index = {row[0]:int(row[1]) for row in csv_reader}

    # Initialize embeddings (words not in glove vocabulary are randomly initialized, padding is zero vector, unknown token is mean embedding)
    embeddings = np.random.uniform(-0.25,0.25,(len(word2index),dim))
    for word in word2index.keys():
        if word in glove.stoi:
            embeddings[word2index[word]] = glove[word]
    embeddings[word2index['<pad>']] = np.zeros(dim)
    embeddings[word2index['<unk>']] = np.mean(embeddings, axis=0)

    return torch.FloatTensor(embeddings)

def cnn_collate(batch):
    """
    Collate function that pads all sequences in the batch.
    """
    data, labels = [], []
    for datapoint in batch:
        data.append(datapoint[0])
        labels.append(datapoint[1])

    data = pad_sequence(data, batch_first=True, padding_value=0)

    return torch.LongTensor(data), torch.tensor(labels)

def rnn_collate(batch):
    """
    Collate function that pads all sequences in the batch and also returns the sentence lengths.
    """
    data, lengths, labels = [], [], []
    for datapoint in batch:
        data.append(datapoint[0])
        lengths.append(len(datapoint[0]))
        labels.append(datapoint[1])

    data = pad_sequence(data, batch_first=True, padding_value=0)

    return [torch.LongTensor(data), lengths], torch.tensor(labels)