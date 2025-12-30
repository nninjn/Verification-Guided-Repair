import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import numpy as np
from .evaluate import idi_test
class Data(Dataset):
    def __init__(self, X, Y):
        self.X, self.Y = X, Y
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
    def __len__(self):
        return len(self.X)

def repair_data_prepare_normal(dataset_loader, normal_num, seed):
    all_features = []
    all_labels = []
    for batch in dataset_loader:
        features, labels = batch
        all_features.append(features.numpy())
        all_labels.append(labels.numpy())
    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.concatenate(all_labels, axis=0)
    rng = np.random.default_rng(seed)
    if len(features_array) < normal_num:
        raise ValueError("Requested normal samples exceed available samples")
    indices = rng.choice(len(features_array), size=normal_num, replace=False)
    selected_features = features_array[indices]
    selected_labels = labels_array[indices]

    result_feature = selected_features
    result_label = selected_labels.reshape(-1, 1)
    result = np.hstack([result_feature, result_label])
    return result, result_feature, result_label, normal_num

def repair_data_prepare_true(model, dataset_loader, normal_num):
    model.eval()
    model.if_sig = True
    device = next(model.parameters()).device
    loader = dataset_loader
    
    correct_features = []
    correct_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device) 
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            
            correct_mask = (predicted == labels).squeeze()
            batch_correct_indices = correct_mask.nonzero(as_tuple=True)[0].tolist()
            
            for idx in batch_correct_indices:
                if len(correct_features) >= normal_num:
                    break
                feature = inputs[idx].cpu().numpy()
                correct_features.append(feature)
                label = labels[idx].cpu().numpy()
                correct_labels.append(label)
                
            if len(correct_features) >= normal_num:
                break
    result_feature = np.array(correct_features[:normal_num])
    result_label = np.array(correct_labels[:normal_num])
    result = np.hstack([result_feature, result_label])
    return result, result_feature, result_label

def dataset_split(full_dataset, seed):
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    # subset_dataset = Subset(full_dataset, full_dataset)
    # return full_dataset, val_dataset, test_dataset
    return train_dataset, val_dataset, test_dataset

def create_segmented_loaders(pair_data, len_markers, batch_size=128):
    if len(len_markers) == 0:
        return []
    markers = len_markers + [pair_data.shape[0]]
    loaders = []
    for i in range(len(markers) - 1):
        start = markers[i]
        end = markers[i+1]
        segment = pair_data[start:end]
        segment_tensor = torch.from_numpy(segment)
        dataset = TensorDataset(segment_tensor)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False
        )
        loaders.append(loader)
    return loaders

def test_acc(model, X_test, y_test):
    if hasattr(model, 'if_sig'):
        model.if_sig = True
    model.eval()
    y = model(X_test)
    preds = torch.where(y > 0.5, 1, 0).squeeze()
    assert preds.shape == y_test.shape, f"error in shape, {preds.shape=} {y_test.shape=}"
    preds = preds.to(y_test.device)
    acc = (preds == y_test).float().mean().item()
    return acc

from dataprocess import data_load
from network import create_model
import torch.nn as nn
from utils.train import train_model
import os
def load_data_model(dataset, sen, X, Y, input_shape, net_name, BATCH_SIZE, device, seed, epochs=10, need_train_model=False):
    model_dir = os.path.join("buggy_model", dataset)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{net_name}_model.pth")
    X_tensor = torch.tensor(X, dtype=torch.float32)
    if dataset == "adult" or dataset == "bank":
        Y_labels = torch.tensor(Y, dtype=torch.float32)
    else:
        Y_labels = torch.argmax(torch.tensor(Y, dtype=torch.float32), dim=1)  # Convert one-hot to class index
    full_dataset = TensorDataset(X_tensor, Y_labels)

    train_dataset, val_dataset, test_dataset = dataset_split(full_dataset=full_dataset, seed=seed)
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)#, shuffle=True)

    model = create_model(net_name, input_size=input_shape[1]).to(device)
    if need_train_model == "True":
        criterion = nn.BCELoss()
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            epochs=epochs)
        torch.save(model.state_dict(), model_path)
    else:
        model.load_state_dict(torch.load(model_path))

    assert len(sen) > 0, "Undefined SA!"
    test_indices = test_dataset.indices
    X_test_np = X[test_indices].astype(np.float32)
    y_test_np = test_dataset.dataset.tensors[1][test_indices].numpy().astype(np.int64)
    A_test_np = X_test_np[:, sen].astype(np.int64)
    train_indices = train_dataset.indices
    X_train_np = X[train_indices].astype(np.float32)
    y_train_np = train_dataset.dataset.tensors[1][train_indices].numpy().astype(np.int64)
    return model, full_loader, X_test_np, y_test_np, A_test_np, X_train_np, y_train_np

def prepare_repair_data(num1, num2, dataset, model, sen, full_loader, X_train_np, y_train_np, BATCH_SIZE, seed, device):
    _, unf_data, unf_num, _ = idi_test(dataset, model, sen, torch.from_numpy(X_train_np), num1, get_data=True, use_base_data=False)
    nor_data, nor_feature, nor_label, nor_num = repair_data_prepare_normal(full_loader, num2, seed)
    un_loader = DataLoader(unf_data, batch_size=BATCH_SIZE)
    nor_loader = DataLoader(nor_data, batch_size=BATCH_SIZE)
    nor_feature_tensor = torch.tensor(nor_feature, dtype=torch.float32).to(device)
    # nor_label_tensor = torch.tensor(nor_label, dtype=torch.long).squeeze(-1).to(device)
    nor_label_tensor = torch.tensor(nor_label, dtype=torch.float32).squeeze(-1).to(device)
    nor_label_np = nor_label_tensor.cpu().numpy()
    return un_loader, unf_data, nor_loader, nor_data, nor_feature, nor_feature_tensor, nor_label_np, unf_num, nor_num

