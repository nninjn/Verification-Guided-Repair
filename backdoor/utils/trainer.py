import os
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
import torch.nn as nn
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
import argparse
from utils.make_dataset import get_standard, get_backdoor
from utils.network import  VGG13_dense, CNN8_dense
import os
import numpy as np
import torch
import torch.nn as nn
import copy
import random
from repair import Repair
import argparse
from utils.prepare import data_model

gpu = 0
device = 'cuda:%s'%gpu if torch.cuda.is_available() else 'cpu'


def train_models(arch, args, seed=2222, epochs=30):
    save_dir = os.path.join(args.save_dir, f'{arch}-{args.act}-{args.attack}/seed{seed}')
    if not os.path.exists(save_dir):
        print('\n new dir <%s>\n' % save_dir)
        os.makedirs(save_dir)
    log_file = open(os.path.join(save_dir[:-8], '%s_%s_%s_log.txt' % (arch, args.set, seed)), 'a')
    log_info = '%s, seed %s, ' % (arch, seed)
    if 'res18_dense' in arch:
        # model = cResNet18()
        pass
    else:
        # model = VGG13_gtsrb_dense('VGG13')
        # model = VGG13_sig('VGG13')
        model = VGG13_dense(act=args.act, vgg_name='VGG13')
        # model = CNN8_dense()

    model.to(device)
    
    nor_dataset = get_standard(root=args.root, set=args.set, process=['std'], num=50000, train=True, seed=23)
    backdoor_dataset = get_backdoor(root=args.root, set=args.set, process=['std'], num=50000, train=True, mode='train', seed=23, attack=args.attack)
    nor_val = get_standard(set=args.set, process=['std'], num=2000, train=False, seed=23)
    bd_val = get_backdoor(set=args.set, process=['std'], num=2000, train=False, mode='ptest', seed=23, attack=args.attack)

    loader_nor = DataLoader(nor_dataset, batch_size=32, shuffle=True, num_workers=0)
    loader_bd = DataLoader(backdoor_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader_nor = DataLoader(nor_val, batch_size=32, shuffle=True, num_workers=0)
    val_loader_bd = DataLoader(bd_val, batch_size=32, shuffle=True, num_workers=0)


    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().cuda()

    #  train normal model
    all_keys = model.state_dict().keys()
    model_keys = [k for k in all_keys if 'probe' not in k]
    acc_hist = []

    if args.set == 'mnist':
        n_epoch = 20
        b_epoch = 5
    else:
        n_epoch = 100
        b_epoch = 20
        

    for e in range(n_epoch):
        acc = std_trainer(model, loader_nor, criterion, optimizer, e)
        acc_hist.append(acc)
        save_model_part(model, model_keys, 'std', out_dir=save_dir, acc_rec=acc_hist)
    
    std_acc = validate(model, val_loader_nor, device=gpu)
    log_info += 'std acc %s, ' % std_acc


    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(param, lr=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss().cuda()


    acc_hist = []
    for e in range(b_epoch):
        acc = std_trainer(model, loader_bd, criterion, optimizer, e)
        acc_hist.append(acc)
        save_model_part(model, model_keys, 'bd', out_dir=save_dir, acc_rec=acc_hist)
    bd_acc = validate(model, val_loader_nor, device=gpu)
    log_info += 'bd acc on nor set %s, ' % bd_acc
    bd_acc = validate(model, val_loader_bd, device=gpu)
    log_info += 'bd attack success rate %s \n ' % bd_acc
    log_file.write(log_info)
    log_file.flush()


def std_trainer(model, loader, criterion, optimizer, e):
    acc = []
    print(loader)
    for images, targets in loader:
        images, targets = images.float().to(device), targets.long().to(device)
        # print(images.shape, targets.shape)
        output = model(images)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        # print(loss.item())
        optimizer.step()
        acc.append(Accuracy(output, targets)[0].cpu().detach().numpy())

    print('end of epoch : {}, top1 train acc : {}'.format(e, np.mean(acc)))
    return np.mean(acc)



def save_model_part(model, state_keys, name, out_dir='./checkpoints', acc_rec=None):
    # save part of the model's weights, like the probe
    save_dir = os.path.join(out_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    state_dict = model.state_dict()
    save_dict = {
        k: v for k, v in state_dict.items()
        if k in state_keys
    }
    if acc_rec is None:
        file_path = os.path.join(save_dir, 'model-best.pt')
        torch.save(save_dict, file_path)
        return
    else:
        if acc_rec[-1] >= max(acc_rec):
            file_path = os.path.join(save_dir, 'model-best.pt')
            torch.save(save_dict, file_path)
            print('old best is {} New best saved {} to {}'.format(max(acc_rec), acc_rec[-1], file_path))
        else:
            return


def validate(model, data_loader, device, per_class=False, nc=10):
    model.eval()
    total,correct = 0, 0
    nc = 10
    class_correct = list(0. for i in range(nc))
    class_total = list(0. for i in range(nc))

    for data, target in data_loader:
        bs = data.size(0)
        data, target = data.float().to(device), target.to(device)
        outputs = model(data)
        outputs = outputs.view(bs, nc)
        _, predicted = torch.max(outputs.data, 1)

        if per_class:
            c = (predicted == target).squeeze()
            for i in range(bs):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        total += target.size(0)
        correct += (predicted == target).sum().item()
    acc_per_class = []

    if per_class:
        for i in range(nc):
            acc_per_class.append(class_correct[i] / class_total[i])
        return acc_per_class

    accuracy = correct / total
    # print("Test Accuracy: {}/ {} = {} ".format(correct, total, accuracy))
    return accuracy



def Accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parse = argparse.ArgumentParser(description='Backdoor Repair')
parse.add_argument('--set', type=str, help='dataset', default='cifar10')
parse.add_argument('--arch', type=str, help='model architecture', default='vgg13_dense')
parse.add_argument('--act', type=str, help='model activation', default='relu')
parse.add_argument('--save_dir', type=str, help='where to save model', default='./checkpoints')
parse.add_argument('--root', type=str, help='where the dataset is', default='./mnist')
parse.add_argument('--attack', type=str, help='attack methods', default='Badnets')
args = parse.parse_args()


if __name__ == '__main__':
    for i in range(1):
        set_random_seed(2022 + i)
        seed = 2022 + i
        train_models(args.arch, args, seed=2022 + i, epochs=100)
