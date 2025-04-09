import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from rnn_model import LSTMClassifier
from y_encoder import encode_y
from rf_train import load_csv
from parameters import *

torch.set_default_dtype(torch.float64)

class ChromagramDataset(Dataset):
    def __init__(self, dataframes):
        self.data = []
        self.labels = []
        for df in dataframes:
            chromagram = df.iloc[:, :-1].values  
            label = list(encode_y(df['y']))
            self.data.append(chromagram)
            self.labels.append(label)
        self.num_classes = len(set(label for labels in self.labels for label in labels))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float64), torch.tensor(self.labels[idx], dtype=torch.long)
    
def collate_fn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # separate source and target sequences
    src_seqs, gt_seqs = zip(*data)

    lengths = [len(seq) for seq in src_seqs]
    # pad src
    padded_seqs = torch.full((len(src_seqs), max(lengths), src_seqs[0].shape[1]), -1)
    for i, seq in enumerate(src_seqs):
        padded_seqs[i, :lengths[i]] = seq
    src_seqs = padded_seqs.double()

    #pad gt
    padded_seqs = torch.full((len(gt_seqs), max(lengths)), -1)
    for i, seq in enumerate(gt_seqs):
        padded_seqs[i, :lengths[i]] = seq
    gt_seqs = padded_seqs.long()

    return src_seqs, gt_seqs, lengths

def eval_model(model, data_loader):
    model.eval()
    correct, total, acc = 0, 0, 0
    with torch.no_grad():
        for data in data_loader:
            inputs, labels, lengths = data
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs, lengths)
            _, predicted = torch.max(outputs.view(-1, model.num_classes), 1)
            total += labels.size(0)
            correct += (predicted == labels.view(-1)).sum().item()
    if total > 0:
        acc = correct / total
    return acc

def rnn_train():
    # Load data from CSV files.
    data_list = load_csv()
    dataset = ChromagramDataset(data_list)
    # Create a DataLoader for batching.
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    # Specify model parameters.
    input_size = 12
    hidden_dim = 50
    num_classes = dataset.num_classes
    num_layers = 2
    use_cuda = torch.cuda.is_available()
    bidirectional = True
    model = LSTMClassifier(input_size, hidden_dim, num_classes, num_layers, use_cuda, bidirectional,
                        dropout=(0.4, 0.0, 0.0))

    if use_cuda:
        model = model.cuda()
    
    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.SGD(model.parameters(), lr=0.005)

    # Training loop
    num_epochs = 10
    model.train()
    print(f'Starting training at {get_time()}...')
    for epoch in tqdm(range(num_epochs)):
        for data, labels, length in data_loader:
            if use_cuda:
                data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(data, length)
            output = output.view(-1, num_classes)
            labels = labels.view(-1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        train_acc = eval_model(model, data_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Training Accuracy: {train_acc:.4f}')

    print(f'Training complete at {get_time()}!')
    torch.save(model.state_dict(), f'{script_dir}/model/lstm_model.pth')
    print(f'Model saved at {script_dir}/model/lstm_model.pth')

if __name__ == "__main__":
    rnn_train()
