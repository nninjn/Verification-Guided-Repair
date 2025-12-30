import numpy as np
from numpy.random import beta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference

def train_model(model, train_loader, val_loader, device, criterion, epochs) -> nn.Module:
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_model_path = "trained_model/best_model.pth"
    model.train()
    for epoch in range(epochs):
        train_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            # print(inputs.shape)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        val_acc, val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        # if(epoch % 50 == 0):
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), best_model_path)
    return model


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.float().unsqueeze(1).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(outputs.cpu().numpy().flatten())

    acc = 100 * correct / total
    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_labels, [p > 0.5 for p in all_probs])
    auc = roc_auc_score(all_labels, all_probs)
    return acc, avg_loss, f1, auc


def evaluate_dp_new(model, X_test, y_test, A_test):
    model.eval()

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output, _, _, _ = model(X_test_cuda)
    pred = np.int64(output.cpu().detach().numpy() > 0.5)
    dp = demographic_parity_difference(y_test, pred, sensitive_features=A_test)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, dp


def evaluate_eo_new(model, X_test, y_test, A_test, testing=False, optimizer=None):
    model.eval()

    # calculate average precision
    X_test_cuda = torch.tensor(X_test).cuda().float()
    output, _, _, _ = model(X_test_cuda)
    pred = np.int64(output.cpu().detach().numpy() > 0.5)
    eo = equalized_odds_difference(y_test, pred, sensitive_features=A_test)
    y_scores = output[:, 0].data.cpu().numpy()
    ap = average_precision_score(y_test, y_scores)

    return ap, eo

