import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from rnn_model import LSTMClassifier
from y_encoder import encode_y
from rf_train import load_csv
from parameters import *
import matplotlib.pyplot as plt

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
            if model.use_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs, lengths)
            predicted = outputs.topk(1, dim=2)[1].squeeze().view(-1)
            labels = labels.view(-1)
            predicted = predicted[labels >= 0]
            labels = labels[labels >= 0]
            total += len(labels)
            correct += (predicted == labels.view(-1)).sum().item()
    if total > 0:
        acc = correct / total
    return acc

def rnn_train():
    # Load data from CSV files.
    train_acc_list = []
    test_acc_list = []
    loss_list = []
    data_list = load_csv()
    train_list, test_list = train_test_split(data_list, test_size=0.05, random_state=42)
    train_dataset, test_datasets = ChromagramDataset(train_list), ChromagramDataset(test_list)
    # Create a DataLoader for batching.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # Specify model parameters.
    use_cuda = torch.cuda.is_available()
    bidirectional = True
    model = LSTMClassifier(input_size, hidden_dim, num_classes, num_layers, use_cuda, bidirectional,
                        dropout=(0.4, 0.0, 0.0))

    if use_cuda:
        model = model.cuda()
    
    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if use_cuda:
        criterion = criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    # Training loop (for demonstration, training on one sample).
    print(f'Starting training at {get_time()}...')
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for data, labels, length in train_loader:
            if use_cuda:
                data, labels = data.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(data, length)
            output = output.view(-1, output.size(2))
            labels = labels.view(-1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        train_acc = eval_model(model, train_loader)
        test_acc = eval_model(model, test_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {loss.item():.4f}, Training Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        # disable dropout on last 10 epochs
        if num_epochs - epoch == 10:
            model.disable_dropout()
        loss_list.append(loss.item())
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

    print(f'Training complete at {get_time()}!')
    torch.save(model.state_dict(), f'{script_dir}/model/lstm_model.pth')
    print(f'Model saved at {script_dir}/model/lstm_model.pth')

    # Plotting the training loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Loss', color='red')
    plt.plot(train_acc_list, label='Train Accuracy', color='blue')
    plt.plot(test_acc_list, label='Test Accuracy', color='green')
    plt.xticks(range(0, num_epochs, 5))
    plt.xlabel('Epochs')
    plt.ylabel('Accurac / Loss')
    plt.title('Loss, Training and Test Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(f'{script_dir}/model/lstm_{hidden_dim}_{num_layers}_{batch_size}_{lr}.png')
    plt.show()

if __name__ == "__main__":
    rnn_train()