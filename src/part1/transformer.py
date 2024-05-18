import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, average_precision_score


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights

class TransformerModel(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, num_classes, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, src, src_mask=None):
        attention_weights = []
        for layer in self.encoder_layers:
            src, attn_weights = layer(src, src_mask)
            attention_weights.append(attn_weights)
        output = self.fc(src)
        return output, attention_weights

def load_train_test(dpath="../../data/ptbdb/"):
    df_train = pd.read_csv(Path(dpath) / 'train.csv', header=None)
    df_test = pd.read_csv(Path(dpath) / 'test.csv', header=None)

    # Train split
    X_train = torch.tensor(df_train.iloc[:, :-1].values, dtype=torch.float32)
    y_train = torch.tensor(df_train.iloc[:, -1].values, dtype=torch.float32)

    # Test split
    X_test = torch.tensor(df_test.iloc[:, :-1].values, dtype=torch.float32)
    y_test = torch.tensor(df_test.iloc[:, -1].values, dtype=torch.float32)

    return X_train, y_train, X_test, y_test


def fit_evaluate(model, train_loader, test_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            targets = targets.unsqueeze(1)  # Convert to tensor
            loss = criterion(outputs, targets)  # Assuming binary classification
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1} Training Loss: {running_loss / len(train_loader)}")

    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for inputs, targets_batch in test_loader:
            outputs, _ = model(inputs)
            predicted = torch.round(torch.sigmoid(outputs))
            total += targets_batch.size(0)
            correct += (predicted == targets_batch.unsqueeze(1)).sum().item()
            predictions.extend(predicted.cpu().numpy())
            targets.extend(targets_batch.cpu().numpy())

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")

    # Calculate AUROC
    auroc = roc_auc_score(targets, predictions)

    # Calculate AUPRC
    auprc = average_precision_score(targets, predictions)

    print(f"AUROC: {auroc}")
    print(f"AUPRC: {auprc}")


def visualize_attention(model, src_len):
    src = ["<sos>"] + [str(i) for i in range(1, src_len-1)] + ["<eos>"]
    trg = ["<sos>"] + ["<eos>"]
    model.eval()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            print(inputs)
            _, attention_weights = model(inputs)
            attention_weights = attention_weights[-1].cpu().numpy().squeeze()

            plt.figure(figsize=(src_len, src_len))
            sns.heatmap(attention_weights, xticklabels=trg, yticklabels=src, annot=True, cbar=False)
            plt.title(f'Attention Heatmap for Example {i+1}')
            plt.xlabel('Target')
            plt.ylabel('Source')
            plt.savefig(f'attention_heatmap_example_{i+1}.png')  # Save figure to a file
            plt.close()  # Close the figure
            print(f"Attention Heatmap for Example {i+1} saved!")    
            if i > 3:
                break  # Only visualize the attention of the first example

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
num_layers = 4
d_model = 187
nhead = 1
dim_feedforward = 512
num_classes = 1
dropout = 0.1
epochs = 10
batch_size = 64

# Load data
X_train, y_train, X_test, y_test = load_train_test()

# Create DataLoader
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize model
model = TransformerModel(num_layers, d_model, nhead, dim_feedforward, num_classes, dropout).to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# Train and evaluate the model
fit_evaluate(model, train_loader, test_loader, optimizer, criterion)

# Visualize attention
visualize_attention(model, X_train.shape[1])
