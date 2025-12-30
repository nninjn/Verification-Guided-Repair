import random
import torch
import torchvision
from utils.network import VGG13_dense, VGG11_dense, VGG16_dense, CNN8_dense
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from utils.PoisonedDataset import PoisonedCifar, PoisonedSVHN, PoisonedGTSRB, PoisonedImageNet, \
                                BlendCifar, BlendSVHN, BlendGTSRB, BlendImageNet, BlendMNIST, PoisonedMNIST
from utils.make_dataset import GTSRB


def data_model(dataset, attack, act='relu'):
    BATCH_SIZE = 64
    data_path = '../../data'
    model_path = './buggy_model'
    if dataset == 'mnist':
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        n_classes = 10
        root = f'{data_path}/MNIST'
        print('-', root)
        clean_test_dataset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=train_transform)                  
    
        if attack == 'Badnets':
            target = 9
            bd_test_dataset = PoisonedMNIST(root=root, train=False, transform=train_transform, trigger_label=9, mode='ptest',
                                    return_true_label=True, corruption_root=None, name=None)
        elif attack == 'Blend':
            target = 0
            print(root)
            bd_test_dataset = BlendMNIST(root=root, train=False, transform=train_transform, trigger_label=0, mode='ptest',
                                return_true_label=True, download=True)
        if act == 'relu':
            poi_dir = f'{model_path}/mnist/cnn-{attack.lower()}.pt'
        else:
            poi_dir = f'{model_path}/mnist/cnn-{act}-{attack.lower()}.pt'
        poisoned_model = CNN8_dense()
        poisoned_model.load_state_dict(torch.load(poi_dir, map_location={'cuda:2': 'cuda:0', 'cuda:5': 'cuda:0'}))  
    elif dataset == 'cifar10' or 'cifar' in dataset:
        train_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
                ])
        mode = 'all poi'
        n_classes = 10
        root = f'{data_path}/CIFAR10'
        clean_test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=train_transform)                     
        if attack == 'Badnets':
            target = 0
            bd_test_dataset = PoisonedCifar(root=root, train=False, transform=train_transform, trigger_label=0, mode='ptest',
                                    return_true_label=True, corruption_root=None, name=None)
        elif attack == 'Blend':
            target = 9
            bd_test_dataset = BlendCifar(root=root, train=False, transform=train_transform, trigger_label=9, mode='ptest',
                                return_true_label=True, corruption_root=None, name=None)
            
        if act == 'relu':
            poi_dir = f'{model_path}/cifar10/cnn-{attack.lower()}.pt'
        else:
            poi_dir = f'{model_path}/cifar10/cnn-{act}-{attack.lower()}.pt'
          
        poisoned_model = VGG13_dense(act=act)
        poisoned_model.load_state_dict(torch.load(poi_dir))       
    elif dataset == 'svhn':
        n_classes = 10
        root = f'{data_path}/SVHN'
        train_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(
                 mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
        clean_test_dataset = torchvision.datasets.SVHN(root=root, split='test', download=True, transform=train_transform)                     
        if attack == 'Badnets':
            target = 7
            bd_test_dataset = PoisonedSVHN(root=root, split='test', transform=train_transform, trigger_label=7, mode='ptest',
                                    return_true_label=True, corruption_root=None, name=None)
        elif attack == 'Blend':
            target = 0
            bd_test_dataset = BlendSVHN(root=root, split='test', transform=train_transform, trigger_label=0, mode='ptest',
                                return_true_label=True, corruption_root=None, name=None)
        if act == 'relu':
            poi_dir = f'{model_path}/svhn/cnn-{attack.lower()}.pt'
        else:
            poi_dir = f'{model_path}/svhn/cnn-{act}-{attack.lower()}.pt'
        poisoned_model = VGG13_dense(act=act)
        poisoned_model.load_state_dict(torch.load(poi_dir, map_location={'cuda:5': 'cuda:0'}))
    elif dataset == 'gtsrb':
        n_classes = 43
        train_transform = transforms.Compose(
                [
                    transforms.Resize([32, 32]),
                    transforms.ToTensor(),
                    # transforms.Normalize([0, 0, 0], [1, 1, 1]),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5]),
                        transforms.Resize([32, 32])
                ])
        root = f'{data_path}/GTSRB'
        clean_test_dataset = GTSRB(root=root, train=False, transform=train_transform)                     
        if attack == 'Badnets':
            target = 4
            bd_test_dataset = PoisonedGTSRB(root=root, transform=train_transform, trigger_label=4, mode='ptest',
                                    return_true_label=True, corruption_root=None, name=None, train=False)
        elif attack == 'Blend':
            target = 6
            bd_test_dataset = BlendGTSRB(root=root, transform=train_transform, trigger_label=6, mode='ptest',
                                return_true_label=True, corruption_root=None, name=None, train=False)
        if act == 'relu':
            poi_dir = f'{model_path}/gtsrb/cnn-{attack.lower()}.pt'
        else:
            poi_dir = f'{model_path}/gtsrb/cnn{act}-{attack.lower()}.pt'
        poisoned_model = VGG11_dense()
        poisoned_model.load_state_dict(torch.load(poi_dir, map_location={'cuda:2': 'cuda:0'}))
        
    elif dataset == 'imagenette':
        BATCH_SIZE = 32
        n_classes = 10
        imagenet_dir = f'{data_path}/imagenette/imagenette2/val'
        train_transform = transforms.Compose(
            [ transforms.Resize(size=256),
                transforms.CenterCrop(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        pt = transforms.Compose(
            [ 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        clean_test_dataset = torchvision.datasets.ImageFolder(imagenet_dir, train_transform)

        if attack == 'Badnets':
            target = 0
            bd_test_dataset = PoisonedImageNet(root=imagenet_dir, transform=train_transform, pattern_transform=pt, 
                                               trigger_label=0, mode='ptest', return_true_label=True)

        elif attack == 'Blend':
            target = 8
            bd_test_dataset = BlendImageNet(root=imagenet_dir, transform=train_transform, pattern_transform=pt, 
                                            trigger_label=8, mode='ptest', return_true_label=True)

        poi_dir = f'{model_path}/imagenette/cnn-{attack.lower()}.pt'
        poisoned_model = VGG16_dense()
        poisoned_model.load_state_dict(torch.load(poi_dir, map_location={'cuda:2': 'cuda:0'}))
        
    return n_classes, target, poisoned_model, clean_test_dataset, bd_test_dataset


def prepare_dataloaders(clean_test_data, poisoned_test_data, 
                        BATCH_SIZE=64, N_clean=1000, N_poi=1000, imagenet=False):
    # if imagenet is True:
    #     BATCH_SIZE = 16 
    total_len = len(clean_test_data)
    usable_indices = list(range(total_len))

    repair_clean_indices = random.sample(usable_indices, N_clean)
    test_clean_indices = list(set(usable_indices) - set(repair_clean_indices))

    repair_poi_indices = random.sample(list(range(len(poisoned_test_data))), N_poi)
    test_poi_indices = list(set(range(len(poisoned_test_data))) - set(repair_poi_indices))

    test_acc_set = Subset(clean_test_data, test_clean_indices)
    test_sr_set = Subset(poisoned_test_data, test_poi_indices)
    repair_acc_set = Subset(clean_test_data, repair_clean_indices)
    repair_sr_set = Subset(poisoned_test_data, repair_poi_indices)
    test_acc_loader = DataLoader(test_acc_set, batch_size=BATCH_SIZE, shuffle=False)
    test_sr_loader = DataLoader(test_sr_set, batch_size=BATCH_SIZE, shuffle=False)

    # clean_data_for_repair_loader = DataLoader(repair_acc_set, batch_size=BATCH_SIZE, shuffle=False)
    # poi_data_for_repair_loader = DataLoader(repair_sr_set, batch_size=BATCH_SIZE, shuffle=False)
    clean_data_for_repair_loader = DataLoader(repair_acc_set, batch_size=N_clean, shuffle=False)
    poi_data_for_repair_loader = DataLoader(repair_sr_set, batch_size=N_poi, shuffle=False)

    return test_acc_set, test_sr_set, repair_acc_set, repair_sr_set, \
            test_acc_loader, test_sr_loader, clean_data_for_repair_loader, poi_data_for_repair_loader
