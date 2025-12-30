import os
import sys
import glob
import logging
import numpy as np
import torch
import torch.nn as nn
import copy
import random
from repair import Repair
import argparse
from utils.prepare import data_model


parse = argparse.ArgumentParser(description='Backdoor Repair')    
parse.add_argument('--dataset', type=str, default='cifar')
parse.add_argument('--attack', type=str, default='Badnets')
parse.add_argument('--interval', type=int, help='interval nums', default=10)
parse.add_argument('--localepoch', type=int, help='epoch for local repair', default=30)
parse.add_argument('--N', type=int, help='poi_input_num for repair', default=1000)
parse.add_argument('--N_clean', type=int, help='clean_input_num for repair', default=1000)
parse.add_argument('--layer_round', type=int, help='repair round for a layer', default=20)
parse.add_argument('--seed', type=int, default=0)
parse.add_argument('--device', type=str, default='cuda:0')
parse.add_argument('--log', action='store_true', help='Do not record log')


args = parse.parse_args() 
print(args)
dataset = args.dataset
attack = args.attack
interval_num = args.interval
local_epoch = args.localepoch
N = args.N
N_clean = args.N_clean
layer_round = args.layer_round
device = args.device

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

log = logging.getLogger('my_unique_logger')
log.setLevel(logging.DEBUG)
log.handlers.clear()
log.propagate = False
if args.log:
    results_dir = f'results/Ours/{N_clean}+{N}'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    file_handler = logging.FileHandler(f"{results_dir}/{args.dataset}_{args.attack}.log", "a")
else:
    file_handler = logging.StreamHandler(sys.stdout)  # or use logging.NullHandler() to suppress
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)
from datetime import datetime
now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
log.info(f"Time: {now_time}")
log.info(f"Repair dataset {args.dataset}")
log.info(f"Repair attack {args.attack}")
log.info(f"Repair num (poisoned) {args.N}")
log.info(f"Repair num (clean) {args.N_clean}")
log.info(f"{args.layer_round=}  {args.interval=}  {args.localepoch=}")
log.info('\t')


# prepare data, model and targeted label
n_classes, target, poisoned_model, clean_test_dataset, bd_test_dataset = data_model(dataset=dataset, attack=attack)


poisoned_model.only_logits = False
ori = copy.deepcopy(poisoned_model)

BATCH_SIZE = 64 if dataset != 'imagenette' else 32
approximate_method = 'backward'

backdoor_reapir = Repair(BATCH_SIZE=BATCH_SIZE, n_classes=n_classes, dataset=dataset, target=target,
                                clean_model=None, poisoned_model=poisoned_model, clean_test_data=clean_test_dataset,
                                poisoned_test_data=bd_test_dataset, approximate_method=approximate_method,
                                N=N, N_clean=N_clean, local_epoch=local_epoch, device=device)

ori_test_acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.test_acc_loader, 'Poisoned Model',)
ori_test_sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.test_sr_loader, 'Poisoned Model')

ori_acc = backdoor_reapir.Test_acc(poisoned_model, backdoor_reapir.clean_data_for_repair_loader, 'Poisoned Model')
ori_sr = backdoor_reapir.Test_SR(poisoned_model, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')

best = N_clean * ori_acc + N * (1 - ori_sr)


for round in range(layer_round):
    backdoor_reapir.repair_data_classification()
    if backdoor_reapir.incor_data_num == 0:
        break
    
    # f_{i+1:l}
    model, check_layer = backdoor_reapir.poisoned_model.split()
    acc, sr = backdoor_reapir.compute_max_effect(check_layer=check_layer, round=round,\
                                            model=model, interval_num=interval_num)
    
    log.info(f'After repair {round + 1} neurons:')
    log.info(f'On repair data acc {acc * 100:.2f}, sr {sr * 100:.2f}')
    if N_clean * acc + N * (1 - sr) > best:
        best = N_clean * acc + N * (1 - sr)
        best_state = {k: v.cpu().clone() for k, v in backdoor_reapir.poisoned_model.state_dict().items()}

backdoor_reapir.poisoned_model.load_state_dict(best_state)
best_net = backdoor_reapir.poisoned_model
acc = backdoor_reapir.Test_acc(best_net, backdoor_reapir.clean_data_for_repair_loader, 'Clean Model')
sr = backdoor_reapir.Test_SR(best_net, backdoor_reapir.poi_data_for_repair_loader, 'Poisoned Model')
# using test set to eval generalization
test_acc = backdoor_reapir.Test_acc(best_net, backdoor_reapir.test_acc_loader, 'Poisoned Model',)
test_sr = backdoor_reapir.Test_SR(best_net, backdoor_reapir.test_sr_loader, 'Poisoned Model')    
RSR = 100 - 100 * sr 
G = 100 - 100 * test_sr

log.info(f"Before Repair:")
log.info(f'On Repair Set: Acc {ori_acc * 100:.2f}, sr {ori_sr * 100:.2f}')
log.info(f'On Test   Set: Acc {ori_test_acc * 100:.2f}, sr {ori_test_sr * 100:.2f}')
log.info(f"After  Repair:")
log.info(f"Loc time = {sum(backdoor_reapir.loc_time.values())}")
log.info(f"Repair time = {sum(backdoor_reapir.repair_time.values())}")
log.info(f'RSR: {RSR:.2f}')
log.info(f'Acc: {test_acc * 100:.2f}')
log.info(f'G: {G:.2f}')
log.info('#' * 100)

