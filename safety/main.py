import os
import torch
import torch.nn as nn
from utils.network import FNN, FNNspilt6
import random
import numpy as np
from repair import Repair
import argparse
import logging
import sys
from utils.prepare import prepare_dataloaders


parse = argparse.ArgumentParser(description='Safety repair')    
parse.add_argument('--interval', type=int, help='interval nums', default=10)
parse.add_argument('--localepoch', type=int, help='epoch for local repair', default=25)
parse.add_argument('--N_vio', type=int, help='number of violated data for repair', default=500)
parse.add_argument('--N_cor', type=int, help='number of correct data for repair', default=500)
parse.add_argument('--layer_round', type=int, help='repair round for a layer', default=20)
parse.add_argument('--seed', type=int, default=0)
parse.add_argument('--device', type=str, default='cuda:0')
parse.add_argument('--model', type=str, default='n21')
parse.add_argument('--property', type=str, default='p2')
parse.add_argument('--log', action='store_true', help='Do not record log')

args = parse.parse_args() 
print(args)
interval_num = args.interval
local_epoch = args.localepoch
N_vio = args.N_vio
N_cor = args.N_cor
layer_round = args.layer_round
device = args.device

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

log = logging.getLogger('my_unique_logger')
log.setLevel(logging.DEBUG)
log.propagate = False

if args.log:
    results_dir = f'results/Ours/{N_cor}+{N_vio}'
    os.makedirs(results_dir, exist_ok=True)
    file_handler = logging.FileHandler(f"{results_dir}/{args.model}_{args.property}.log", "a")
else:
    file_handler = logging.StreamHandler(sys.stdout)  # or use logging.NullHandler() to suppress
file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)
from datetime import datetime
now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
log.info(f"Time: {now_time}")
log.info(f"Repair net {args.model}")
log.info(f"Repair property {args.property}")
log.info(f"Repair num (violated) {args.N_vio}")
log.info(f"Repair num (clean) {args.N_cor}")
log.info(f"{args.layer_round=}  {args.interval=}  {args.localepoch=}")
log.info('\t')

buggy_nn = FNN()
n_classes = 5

model_path = f'model/{args.model}.pth'
buggy_nn.load_state_dict(torch.load(model_path))
buggy_nn.eval()


BATCH_SIZE = 256
path = f'data/{args.model}-{args.property}'
correct_data = torch.load(f'{path}/drawdown.pt').to(device)
correct_data_test = torch.load(f'{path}/drawdown_test.pt').to(device)
mis_data = torch.load(f'{path}/counterexample.pt').to(device)
mis_data_test = torch.load(f'{path}/counterexample_test.pt').to(device)
print(mis_data_test.shape, correct_data_test.shape)


cor_data, vio_data = prepare_dataloaders(correct_data, mis_data, N_cor=N_cor, N_vio=N_vio)
del correct_data, mis_data

approximate_method = 'backward (CROWN)'


safety_reapir = Repair(BATCH_SIZE=BATCH_SIZE, n_classes=n_classes, property=args.property, buggy_model=buggy_nn, \
                    cor_data=cor_data, test_cor_data=correct_data_test, vio_data=vio_data, test_vio_data=mis_data_test, 
                    approximate_method=approximate_method,  local_epoch=local_epoch, device=device)

# 模型原始性能
ori_vr_Dm = safety_reapir.Test_VR(buggy_nn, vio_data)
ori_vr_Dc = safety_reapir.Test_VR(buggy_nn, correct_data_test)
ori_vr_Dg = safety_reapir.Test_VR(buggy_nn, mis_data_test)

# 对模型进行drawdown测试
cor_vr = safety_reapir.Test_VR(buggy_nn, cor_data)
vio_vr = safety_reapir.Test_VR(buggy_nn, vio_data)

buggy_nn.only_logits = False

record = []
record.append([100 - cor_vr, vio_vr])
best = len(cor_data) * (100 - cor_vr) + len(vio_data) * (100 - vio_vr)
print(best, 'hhh')
check_layer = 5
check_layer = 6

analyze_neuron_num = 50
split_model = buggy_nn.split(check_layer)


for round in range(layer_round):
    safety_reapir.repair_data_classification()
    if len(safety_reapir.unsat_data) == 0:
        print('Model is perfect, stop repair!')
        print('--' * 20)
        break
        
    print('-' * 100, 'Round: ', round)
    normal_vio, mis_vio = safety_reapir.compute_max_effect(check_layer=check_layer, round=round,\
                                                model=split_model, analyze_neuron_num=analyze_neuron_num, N=N_vio, interval_num=5)


print('loc time = ', safety_reapir.loc_time, 'repair time = ', safety_reapir.repair_time)
# torch.save(safety_reapir.buggy_model.state_dict(), 'n21/model/best model.pth')

generalization = 100 - safety_reapir.Test_VR(buggy_nn, mis_data_test)
drawdown = safety_reapir.Test_VR(buggy_nn, correct_data_test)
rsr = 100 - safety_reapir.Test_VR(buggy_nn, vio_data)

log.info(f"Layer: {check_layer}")
log.info(f"Before  Repair:")
log.info(f'VR on Dm: {ori_vr_Dm:.2f}')
log.info(f'VR on Dc: {ori_vr_Dc:.2f}')
log.info(f'VR on Dg: {ori_vr_Dg:.2f}')
log.info(f"After  Repair:")
log.info(f'RSR: {rsr:.2f}')
log.info(f'D: {drawdown:.2f}')
log.info(f'G: {generalization:.2f}')
log.info(f"Loc time = {sum(safety_reapir.loc_time.values())}")
log.info(f"Repair time = {sum(safety_reapir.repair_time.values())}")




ori_nn = FNN()
ori_nn.load_state_dict(torch.load(f'model/{args.model}.pth'))
ori_nn.eval()
ori_nn = ori_nn.to(device)

num1 = 2
num2 = 1
normed_2 = torch.load('prodrawdown/n%d%d_drawdown_test_2.pt' % (num1, num2))
normed_3 = torch.load('prodrawdown/n%d%d_drawdown_test_3.pt' % (num1, num2))
normed_7 = torch.load('prodrawdown/n%d%d_drawdown_test_7.pt' % (num1, num2))

normed_2 = normed_2.to(device)
normed_3 = normed_3.to(device)
normed_7 = normed_7.to(device)
ori_output = ori_nn(normed_2)
ori_pred_y = torch.max(ori_output.cpu(), 1)[1].numpy()

# p2: class 0 score is not minimal
train_output = safety_reapir.buggy_model(normed_2.to(device))
pred_y_2 = torch.max(train_output.cpu(), 1)[1].numpy()
ori_acc_2 = 0
repair_acc_2 = 0
ori_acc_2 += np.sum(ori_pred_y != 0)
repair_acc_2 += (pred_y_2 != 0).sum()
log.info('Test on p2 space  :   ori acc: {}, repair acc: {}, num: {}, bzd: {}'.format(ori_acc_2, repair_acc_2, normed_2.shape[0], repair_acc_2 / normed_2.shape[0]))
if repair_acc_2 / normed_2.shape[0] < 0.5:
    print(ori_output[0:10])
    print(train_output[0:10])

ori_output = ori_nn(normed_3)
ori_pred_y = torch.min(ori_output.cpu(), 1)[1].numpy()
train_output = safety_reapir.buggy_model(normed_3.to(device))
pred_y_3 = torch.min(train_output.cpu(), 1)[1].numpy()
ori_acc_3 = 0
repair_acc_3 = 0
ori_acc_3 += np.sum(ori_pred_y != 0)
repair_acc_3 += (pred_y_3 != 0).sum()
log.info('Test on p3 space  :   ori acc: {}, repair acc: {}, num: {}, bzd: {}'.format(ori_acc_3, repair_acc_3, normed_3.shape[0], repair_acc_3 / normed_3.shape[0]))
if repair_acc_3 / normed_3.shape[0] < 0.5:
    print(ori_output[0:10])
    print(train_output[0:10])

ori_output = ori_nn(normed_7)
ori_pred_y = torch.min(ori_output.cpu(), 1)[1].numpy()
train_output = safety_reapir.buggy_model(normed_7.to(device))
pred_y_7 = torch.min(train_output.cpu(), 1)[1].numpy()
ori_acc_7 = 0
repair_acc_7 = 0
ori_acc_7 += normed_7.shape[0] - np.sum(ori_pred_y == 3) - np.sum(ori_pred_y == 4)
repair_acc_7 += normed_7.shape[0] - (pred_y_7 == 3).sum() - (pred_y_7 == 4).sum()
log.info('Test on p7 space  :   ori acc: {}, repair acc: {}, num: {}, bzd: {}'.format(ori_acc_7, repair_acc_7, normed_7.shape[0], repair_acc_7 / normed_7.shape[0]))
if repair_acc_7 / normed_7.shape[0] < 0.5:
    print(ori_output[0:10])
    print(train_output[0:10])
log.info('Test on all space :   ori acc: {}, repair acc: {}, num: {}, bzd: {}'.format(\
    ori_acc_2 + ori_acc_3 + ori_acc_7,\
    repair_acc_2 + repair_acc_3 + repair_acc_7,\
    normed_2.shape[0] + normed_3.shape[0] + normed_7.shape[0],\
    (repair_acc_2 + repair_acc_3 + repair_acc_7) / (normed_2.shape[0] + normed_3.shape[0] + normed_7.shape[0])))

log.info('#' * 100)