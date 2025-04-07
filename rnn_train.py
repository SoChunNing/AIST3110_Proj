import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
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
            label = pd.DataFrame(encode_y(df['y']))
            self.data.append(torch.tensor(chromagram))
            self.labels.append(torch.tensor(label, dtype=torch.long))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
        
def rnn_train():
    # Load data from CSV files.
    data_list = load_csv()
    dataset = ChromagramDataset(data_list)
    # Create a DataLoader for batching.
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    # Specify model parameters.
    input_size = 12
    hidden_dim = 50
    num_classes = 10  # Adjust this to match your number of chord labels.
    num_layers = 1
    use_cuda = torch.cuda.is_available()
    bidirectional = True

    model = LSTMClassifier(input_size, hidden_dim, num_classes, num_layers, use_cuda, bidirectional,
                        dropout=(0.4, 0.0, 0.0))

    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop (for demonstration, training on one sample).
    num_epochs = 10
    model.train()
    print(f'Starting training at {get_time()}...')
    for epoch in tqdm(range(num_epochs)):
        for data, labels in data_loader:
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}')

    print(f'Training complete at {get_time()}!')
