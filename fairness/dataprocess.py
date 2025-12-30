import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
sys.path.append("../")

def update_dataset_pth(cfgs):
    dataset = cfgs["dataset"]
    cfgs["val_x_path"] = "data/PGD_dataset/{}/x_val.npy".format(dataset)
    cfgs["val_y_path"] = "data/PGD_dataset/{}/y_val.npy".format(dataset)
    cfgs["train_x_path"] = "data/PGD_dataset/{}/x_train.npy".format(dataset)
    cfgs["train_y_path"] = "data/PGD_dataset/{}/y_train.npy".format(dataset)
    cfgs["test_x_path"] = "data/PGD_dataset/{}/x_test.npy".format(dataset)
    cfgs["test_y_path"] = "data/PGD_dataset/{}/y_test.npy".format(dataset)
    cfgs["constraint_path"] = "data/PGD_dataset/{}/constraint.npy".format(dataset)
    return cfgs

def adult():
    """
    Prepare the data of dataset Adult Census Income
    :return: X, Y, input shape and number of classes
    """
    configs = {"dataset": "adult"}
    configs = update_dataset_pth(configs)
    nb_classes = 2
    input_shape = (None, 12)
    sen = [0, 6, 7]
    configs['input_dim'] = input_shape[1]
    configs['output_dim'] = nb_classes
    
    data_dir = f"data/PGD_dataset/{configs['dataset']}"
    os.makedirs(data_dir, exist_ok=True)

    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))

    x_val = np.load(os.path.join(data_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    X = np.vstack((x_train, x_val, x_test))
    Y = np.hstack((y_train, y_val, y_test))

    return X, Y, input_shape, nb_classes, sen

def bank_data():
    """
    Prepare the data of dataset Bank Marketing
    :return: X, Y, input shape and number of classes
    """
    configs = {"dataset": "bank"}
    configs = update_dataset_pth(configs)
    nb_classes = 2
    input_shape = (None, 16)
    sen = [0]
    configs['input_dim'] = input_shape[1]
    configs['output_dim'] = nb_classes
    data_dir = f"data/PGD_dataset/{configs['dataset']}"
    os.makedirs(data_dir, exist_ok=True)

    x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))

    x_val = np.load(os.path.join(data_dir, 'x_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))

    x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    X = np.vstack((x_train, x_val, x_test))
    Y = np.hstack((y_train, y_val, y_test))

    return X, Y, input_shape, nb_classes, sen

def census_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./processed_data/census", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 13)
    nb_classes = 2
    sensitive_indices = [9, 8, 1]
    return X, Y, input_shape, nb_classes, sensitive_indices

def compas_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./processed_data/compas", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 12)
    nb_classes = 2
    sensitive_indices = [3, 2, 1]
    return X, Y, input_shape, nb_classes, sensitive_indices

def credit_data():
    """
    Prepare the data of dataset German Credit
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./processed_data/credit_sample", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 20)
    nb_classes = 2
    sensitive_indices = [13, 9]
    return X, Y, input_shape, nb_classes, sensitive_indices

def default_data():
    """
    Prepare the data of dataset Default Credit
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./processed_data/default", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 23)
    nb_classes = 2
    sensitive_indices = [5, 2]
    return X, Y, input_shape, nb_classes, sensitive_indices

def diabetes_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./processed_data/diabetes", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 8)
    nb_classes = 2
    sensitive_indices = [8]
    return X, Y, input_shape, nb_classes, sensitive_indices

def heart_data():
    """
    Prepare the data of dataset Heart Disease
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./processed_data/heart_disease", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    input_shape = (None, 13)
    nb_classes = 2
    sensitive_indices = [2, 1]
    return X, Y, input_shape, nb_classes, sensitive_indices

def students_data():
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = []
    Y = []
    i = 0

    with open("./processed_data/students", "r") as ins:
        for line in ins:
            line = line.strip()
            line1 = line.split(',')
            if (i == 0):
                i += 1
                continue
            # L = map(int, line1[:-1])
            L = [int(i) for i in line1[:-1]]
            X.append(L)
            if int(line1[-1]) == 0:
                Y.append([1, 0])
            else:
                Y.append([0, 1])
    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)

    input_shape = (None, 32)
    nb_classes = 2
    sensitive_indices = [3, 2]
    return X, Y, input_shape, nb_classes, sensitive_indices


def meps15_data():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from aif360.datasets.meps_dataset_panel19_fy2015 import MEPSDataset19
    cd = MEPSDataset19()
    le = LabelEncoder()
    df = pd.DataFrame(cd.features)
    df[0] = pd.cut(df[0],9, labels=[i for i in range(1,10)])
    df[2] = pd.cut(df[2],10, labels=[i for i in range(1,11)])
    df[3] = pd.cut(df[3],10, labels=[i for i in range(1,11)])
    df = df.astype('int').drop(columns=[10])
    df[4] = le.fit_transform(df[4])
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = np.array(df.to_numpy(), dtype=int)
    Y = np.array(cd.labels, dtype=int)
    Y = np.eye(2)[Y.reshape(-1)]
    Y = np.array(Y, dtype=int)
    input_shape = (None, len(X[0]))
    nb_classes = 2
    sensitive_indices = [1, 2, 10]
    return X, Y, input_shape, nb_classes, sensitive_indices

def meps16_data():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from aif360.datasets.meps_dataset_panel21_fy2016 import MEPSDataset21
    cd = MEPSDataset21()
    le = LabelEncoder()
    df = pd.DataFrame(cd.features)
    df[0] = pd.cut(df[0],9, labels=[i for i in range(1,10)])
    df[2] = pd.cut(df[2],10, labels=[i for i in range(1,11)])
    df[3] = pd.cut(df[3],10, labels=[i for i in range(1,11)])
    df = df.astype('int').drop(columns=[10])
    df[4] = le.fit_transform(df[4])
    """
    Prepare the data of dataset Census Income
    :return: X, Y, input shape and number of classes
    """
    X = np.array(df.to_numpy(), dtype=int)
    Y = np.array(cd.labels, dtype=int)
    Y = np.eye(2)[Y.reshape(-1)]
    Y = np.array(Y, dtype=int)
    input_shape = (None, len(X[0]))
    nb_classes = 2
    sensitive_indices = [1, 2, 10]
    return X, Y, input_shape, nb_classes, sensitive_indices



DATASET_MAP = {
    "adult": adult,
    "bank": bank_data,
    "census": census_data,
    "compas": compas_data,
    "credit":credit_data,
    "default":default_data,
    "diabetes":diabetes_data,
    "heart":heart_data,
    "students":students_data,
    "meps15":meps15_data,
    "meps16":meps16_data
}

def data_load(dataset_name):
    return DATASET_MAP[dataset_name]()