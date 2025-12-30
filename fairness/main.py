import os
import sys
import time
import torch
import random
import argparse
import numpy as np
from repair import Repair
from utils.utils import *
from utils.data_config import *
from network import create_model
from dataprocess import data_load
from utils.evaluate import evaluation
import logging
import os
from datetime import datetime


parse = argparse.ArgumentParser(description='fairness')  
parse.add_argument('--dataset', type=str, default="census")#compas
parse.add_argument('--SA', type=str, nargs='*', default=['race'])#'sex','age'
parse.add_argument('--interval', type=int, help='interval nums', default=10)
parse.add_argument('--N_normal', type=int, help='clean_input_num for repair', default=200)
parse.add_argument('--N_unfair', type=int, help='unf_input_num for repair', default=200)
parse.add_argument('--layer_round', type=int, help='repair round for a layer', default=20)
parse.add_argument('--local_epoch', type=int, help='local epoch for train', default=30)
parse.add_argument('--device', type=str, default='cuda:1')
parse.add_argument('--seed', type=int, default=0)
parse.add_argument('--log', action='store_true', help='whether using log')


args = parse.parse_args() 
repair_layer = 1
random.seed(args.seed)
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)
BATCH_SIZE = 128
device = args.device

net = {'compas': 'net1', 'meps15': 'net1', 'census': 'net2', 'bank': 'net2'}[args.dataset]
approximate_method = 'backward'
# print(args)

log = logging.getLogger(f'{args.dataset}_{args.SA}')
log.setLevel(logging.DEBUG)
log.handlers.clear()
log.propagate = False
if args.log:
    log_dir = f"results/Ours/{args.N_normal}+{args.N_unfair}*2"
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{args.dataset}_{'_'.join(args.SA)}.log" if args.SA else f"{log_dir}/{args.dataset}.log"
    file_handler = logging.FileHandler(log_path, mode='a')
else:
    file_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.DEBUG)
log.addHandler(file_handler)


now = datetime.now()
now_time = now.strftime("%Y-%m-%d %H:%M:%S")
log.info(f"Time: {now_time}")
log.info(f"Repair Dataset: {args.dataset}")
log.info(f"Sensitive Attributes: {', '.join(args.SA)}")
log.info(f"Data Config: Normal Samples={args.N_normal}, Unfair Samples={args.N_unfair}")
log.info(f"{args.layer_round=} {args.interval=} {args.local_epoch=}") 
log.info('\t')


X, Y, input_shape, nb_classes, sen = data_load(dataset_name=args.dataset)

dataset = globals()[args.dataset]
sen = [dataset.sensitive_feature[sa] for sa in args.SA]
model, full_loader, X_test_np, y_test_np, _, X_train_np, y_train_np = load_data_model(args.dataset, sen, X, Y,\
            input_shape, net, BATCH_SIZE, args.device, args.seed)


ori_acc, ori_idi_per, ori_idi_per_data, uf_d_t, uf_d_t2 = evaluation(model, args.dataset, X_test_np, y_test_np, sen, args.device, verbose=False, unfair_data_tensor=None, unfair_data_tensor2=None)

unf_loader, unf_data, nor_loader, nor_data, nor_inputs, nor_feature_tensor, nor_label,\
    unf_num, nor_num = prepare_repair_data(args.N_unfair, args.N_normal, args.dataset, model, sen,\
                            full_loader, X_train_np, y_train_np, BATCH_SIZE, args.seed, device)

args.N_unfair = unf_num

fair = Repair(BATCH_SIZE=BATCH_SIZE, model=model, nor_data=nor_inputs, un_data=unf_data, \
              approximate_method=approximate_method, interval_times=args.interval, local_epoch=args.local_epoch, device=device)

fair.incor_data_num = unf_num


best, ori_nor, ori_unf = fair.total_test(nor_feature_tensor, nor_label, unf_loader, nor_num, unf_num, fair.model, verbose=False)

split_model, check_layer = fair.model.split()
current_layer = split_model.layers[0]
neuron_num = current_layer.in_features

for round_idx in range(args.layer_round):
    if fair.incor_data_num == 0:
        break
    split_model.if_sig = False
    fair.repair_data_classification()
    fair.compute_max_effect(check_layer=check_layer, model=split_model, neuron_num=neuron_num, \
                            nor_feature=nor_inputs)
    split_model.if_sig = True

    current_score, nor_results, unf_results = fair.total_test(nor_feature_tensor, nor_label, unf_loader, nor_num, unf_num, fair.model)
    log.info(f"After repair {round_idx+1} neuron:")
    log.info(f'Normal data acc: {nor_results*100:.2f}% → Unfair date IDI: {unf_results*100:.2f}%')
    if current_score > best:
        best = current_score
        best_state = {k: v.cpu().clone() for k, v in fair.model.state_dict().items()}

final_model = create_model(net, input_size=input_shape[1]).to(args.device)
final_model.load_state_dict(best_state)
current_score, nor_results, unf_results = fair.total_test(nor_feature_tensor, nor_label, unf_loader, nor_num, unf_num, final_model, verbose=False)

acc, idi_per, idi_per_data = evaluation(final_model, args.dataset, X_test_np, y_test_np, sen, device, verbose=False, unfair_data_tensor=uf_d_t, unfair_data_tensor2=uf_d_t2)
print(f'repair time: {fair.repair_time}')

log.info(f"Before Repair:")
log.info(f"Acc: {ori_acc*100:.2f}%")
log.info(f"D_g: {100-ori_idi_per_data*100:.2f}%")
log.info(f"D_s: {100-ori_idi_per*100:.2f}%")
log.info(f"RSR: {100-ori_unf*100:.2f}%")
log.info(f"After Repair:")
log.info(f"Acc: {acc*100:.2f}%")
log.info(f"D_g: {100-idi_per_data*100:.2f}%")
log.info(f"D_s: {100-idi_per*100:.2f}%")
log.info(f"RSR: {100-unf_results*100:.2f}%")
log.info(f"Fault Localization Time: {fair.loc_time:.2f}s")
log.info(f"Repair Synthesis Time: {fair.repair_time:.2f}s")
log.info(f"Total Time: {fair.loc_time + fair.repair_time:.2f}s")
log.info('#' * 100 + '\n')