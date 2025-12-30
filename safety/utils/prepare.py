import random
from torch.utils.data import DataLoader, Dataset, Subset


def prepare_dataloaders(correct_data, mis_data, N_cor=1000, N_vio=1000):
    
    cor_indices = list(range(len(correct_data)))
    indices1 = random.sample(cor_indices, N_cor)

    mis_indices = list(range(len(mis_data)))
    indices2 = random.sample(mis_indices, N_vio)


    return  correct_data[indices1], mis_data[indices2]
