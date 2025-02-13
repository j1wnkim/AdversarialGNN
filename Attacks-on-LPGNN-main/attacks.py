import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import Enum

    
class AttackMode(Enum):
    ADDNODES = 'addNodes'
    FLIPNODES = 'flipNodes'
    INFERENCE = 'inference'
    POISON = 'poison'
    SHADOW = "shadow"
    

def prepare_data(data):
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, optimizer, epochs):
    
    criterion = nn.BCELoss()
    model.train()
    print("The model is now training")
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for features, labels in train_loader:
            optimizer.zero_grad()
            output = model(features.float())
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            predicted = output.round()
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

        
def test_model(model, test_loader):
    
    criterion = nn.BCELoss()
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features.float())
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
            predicted = outputs.round()
            correct += (predicted == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')

    
def predict(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data.float())
        pred = output.round()
        return pred
    
    
def average_feature_difference(a, b):
    diff = torch.abs(a - b)
    average_diff = torch.mean(diff)
    return average_diff.item()
    