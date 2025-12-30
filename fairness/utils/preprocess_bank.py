import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import os

def preprocess_bank_data(data_path, out_path):
    # 设定随机种子使输出稳定
    torch.manual_seed(42)
    np.random.seed(42)

    # 加载银行数据集
    df = pd.read_csv(data_path, sep=";", encoding='latin-1')

    # 用最频繁的值填充缺失值
    df.replace('unknown', np.nan, inplace=True)
    for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # 将分类属性编码为整数
    data = df.values
    list_index_cat = [1, 2, 3, 4, 6, 7, 8, 10, 15, 16]
    for i in list_index_cat:
        vocab = np.unique(data[:, i])
        # 创建分类值到整数索引的映射
        mapping = {label: idx for idx, label in enumerate(vocab)}
        data[:, i] = np.array([mapping[str(item)] for item in data[:, i]], dtype=np.int64)
    data = data.astype(np.int32)

    # 对数值属性进行分箱处理
    bins_age = [15, 25, 45, 65, 120]
    bins_balance = [-1e4] + [np.percentile(data[:, 5], percent, axis=0) for percent in [25, 50, 75]] + [2e5]
    bins_day = [0, 10, 20, 31]
    bins_month = [-1, 2, 5, 8, 11]
    bins_duration = [-1.0] + [np.percentile(data[:, 11], percent, axis=0) for percent in [25, 50, 75]] + [6e3]
    bins_campaign = [0.0] + [np.percentile(data[:, 12], percent, axis=0) for percent in [25, 50, 75]] + [1e2]
    bins_pdays = [-10.0] + [np.percentile(data[:, 13], percent, axis=0) for percent in [25, 50, 75]] + [1e3]
    bins_previous = [-1.0] + [np.percentile(data[:, 14], percent, axis=0) for percent in [25, 50, 75]] + [3e2]
    list_index_num = [0, 5, 9, 10, 11, 12, 13, 14]
    list_bins = [bins_age, bins_balance, bins_day, bins_month, bins_duration, bins_campaign, bins_pdays, bins_previous]
    for index, bins in zip(list_index_num, list_bins):
        data[:, index] = np.digitize(data[:, index], bins, right=True)

    # 划分数据为训练集、验证集和测试集
    X = data[:, :-1]
    y = data[:, -1]
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=42)

    # 设置每个属性的约束条件
    constraint = np.vstack((X.min(axis=0), X.max(axis=0))).T

    # 对于银行营销数据，年龄（索引0）是受保护属性
    protected_attribs = [0]

    # 保存预处理后的数据
    np.save(f'{out_path}/constraint.npy', constraint)
    np.save(f'{out_path}/x_train.npy', X_train)
    np.save(f'{out_path}/y_train.npy', y_train)
    np.save(f'{out_path}/x_val.npy', X_val)
    np.save(f'{out_path}/y_val.npy', y_val)
    np.save(f'{out_path}/x_test.npy', X_test)
    np.save(f'{out_path}/y_test.npy', y_test)
    np.save(f'{out_path}/protected_attribs.npy', protected_attribs)

    return X_train, X_val, y_train, y_val

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from network import create_model  # 假设存在create_model函数
from utils.train import train_model  # 假设存在train_model函数

def preprocess_bank_data(data_path, sen, net_name, model_path, batch_size, device, seed):
    # 设定随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载数据并进行预处理（省略详细步骤，保持与原函数相同）
    # 这里仅保留关键步骤以简化示例
    df = pd.read_csv(data_path, sep=";", encoding='latin-1')
    # ... [数据预处理步骤：填充缺失值、编码分类变量、分箱等] ...
    # 假设预处理后得到X_train, X_val, X_test, y_train, y_val, y_test

    # 转换数据类型为float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # 合并训练和验证集用于完整训练数据集
    X_train_all = np.concatenate([X_train, X_val], axis=0)
    y_train_all = np.concatenate([y_train, y_val], axis=0)

    # 创建TensorDataset和DataLoader
    train_all_dataset = TensorDataset(
        torch.tensor(X_train_all, dtype=torch.float32),
        torch.tensor(y_train_all, dtype=torch.long)
    )
    full_loader = DataLoader(train_all_dataset, batch_size=batch_size, shuffle=True)

    # 创建模型
    input_shape = X_train_all.shape[1]
    model = create_model(net_name, input_size=input_shape).to(device)

    # 训练或加载模型
    if model_path == 'none':
        # 注意：二分类任务可能需要调整损失函数
        criterion = nn.BCEWithLogitsLoss()  # 假设模型输出未经过sigmoid
        # 创建训练和验证DataLoader
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)  # BCE需要float标签
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            criterion=criterion,
            epochs=100
        )
    else:
        model.load_state_dict(torch.load(model_path))

    # 提取测试数据和训练数据
    X_test_np = X_test.astype(np.float32)
    y_test_np = y_test.astype(np.int64)
    A_test_np = X_test[:, sen].astype(np.int64)  # 假设sen是受保护属性索引列表

    X_train_np = X_train_all.astype(np.float32)
    y_train_np = y_train_all.astype(np.int64)

    return model, full_loader, X_test_np, y_test_np, A_test_np, X_train_np, y_train_np
