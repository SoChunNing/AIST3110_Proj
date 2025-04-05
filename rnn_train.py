import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rnn_model import LSTMClassifier

class ChromagramDataset(Dataset):
    def __init__(self, df_list):
        self.data = []
        self.labels = []
        for df in df_list:
            # Assume the DataFrame contains chromagram data in the first 12 columns
            # and a 'label' column for the corresponding label.
            chroma = df.iloc[:, :12].values  # shape: (sequence_length, 12)
            # Here we assume a single label for the whole sequence
            # Alternatively, you can also have per-time-step labels.
            label = df['label'].iloc[0]  
            self.data.append(chroma)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert the chromagram and label into PyTorch tensors.
        # For an RNN, the input shape should be (sequence_length, num_features).
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def rnn_train(csv_file):
    features = df.iloc[:, :12].values  # shape: (sequence_length, 12)
    label_value = df['label'].iloc[0]    # Assuming one label for the entire sequence
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
    for epoch in range(num_epochs):
        for batch_data, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    print("Training complete!")
