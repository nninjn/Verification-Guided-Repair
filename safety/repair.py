import os
import copy
import time
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm


class Repair():

    def __init__(self, BATCH_SIZE, n_classes, property, buggy_model, cor_data, test_cor_data, vio_data, test_vio_data,  
                    approximate_method, local_epoch, device) -> None:

        self.BATCH_SIZE = BATCH_SIZE
        self.property = property
        self.buggy_model = buggy_model
        self.buggy_model.eval()
        self.device = device
        self.buggy_model = self.buggy_model.to(self.device)

        self.cor_data = cor_data
        self.vio_data = vio_data
        self.all_data = torch.cat([self.cor_data, self.vio_data], dim=0)

        self.test_cor_data = test_cor_data
        self.test_vio_data = test_vio_data
        
        self.n_classes = n_classes
        self.approximate_method = approximate_method
        self.local_epoch = local_epoch

        self.repair_time = {}
        self.loc_time = {}

        if self.property == 'p2':
            self.constraint = [1, 2, 3, 4] # argmin = 0
        elif self.property == 'p7':
            self.constraint = [3, 4] # argmin != 3, 4
        elif self.property == 'p8':
            self.constraint = [2, 3, 4] # argmin = 0, 1
        else:
            raise ValueError(f"Unsupported property: {self.property}")
        
        if self.property in ['p2']:
            self.conj = True 
        else:
            self.conj = False

        self.constraint = torch.tensor(self.constraint).to(self.device)
         

    def Test_VR(self, nn_test, inputs):
        nn_test.eval()
        nn_test.only_logits = True
        vio = 0

        outputs = nn_test(inputs)
        pred_y = torch.min(outputs.cpu(), 1)[1].numpy()
        # print(outputs)
        if self.property == 'p2':
            vio += (pred_y == 1).sum() + (pred_y == 2).sum() + (pred_y == 3).sum() + (pred_y == 4).sum()
        elif self.property == 'p7':
            vio += (pred_y == 3).sum() + (pred_y == 4).sum()
        elif self.property == 'p8':
            vio += (pred_y == 2).sum() + (pred_y == 3).sum() + (pred_y == 4).sum()
        vr =  vio / inputs.shape[0] * 100
        print('-' * 20, f'VR = {vio} / {inputs.shape[0]} = {vr} ',)
        # print(pred_y)
        if vr == 0.4:
            vio = (pred_y != 0)
            print(outputs[vio])
        return vr


    def repair_data_classification(self):
        self.buggy_model.only_logits = True

        output = self.buggy_model(self.all_data)
        pred = torch.min(output, dim=1)[1]

        unsat = (pred.unsqueeze(1) == self.constraint).any(dim=1)
        self.sat_data = self.all_data[~unsat]
        self.unsat_data = self.all_data[unsat]

        print('cor data : ', len(self.sat_data))
        print('incor data : ', len(self.unsat_data))
        self.sat_data_num = len(self.sat_data)
        self.unsat_data_num = len(self.unsat_data)


    def get_C_loc(self, logits):
        # note that the models for ACAS Xu system use min to get the predicted label
        # first select the conjunctive group that is closest to being satisfied, then count the constraints within it that are violated.
        pred_y = torch.min(logits, 1)[1].to(self.device)
        C = torch.zeros(size=(self.unsat_data_num, 1, self.n_classes), device=self.device)
        if self.property == 'p2':
            logit_c0 = logits[:, 0].unsqueeze(1)
            violation_mask = (logits < logit_c0)
            C[:, 0, 0] = -violation_mask.sum(dim=1)
            C[:, 0, 1:] = violation_mask[:, 1:].float()
            
        elif self.property == 'p7':
            best_value, bset_cls = torch.min(logits[:, [0, 1, 2]], dim=1)
            violation_mask = logits[:, [3, 4]] < best_value.unsqueeze(1)
            C[torch.arange(self.unsat_data_num), 0, bset_cls] = -violation_mask.sum(dim=1).float().to(self.device)
            C[:, 0, [3, 4]] = violation_mask.float().to(self.device)

        elif self.property == 'p8':
            best_value, bset_cls = torch.min(logits[:, [0, 1]], dim=1)
            violation_mask = logits[:, [2, 3, 4]] < best_value.unsqueeze(1)
            C[torch.arange(self.unsat_data_num), 0, bset_cls] = -violation_mask.sum(dim=1).float().to(self.device)
            C[:, 0, [2, 3, 4]] = violation_mask.float().to(self.device)
        V_ori = (logits * C[:, 0, :]).sum(dim=-1)
        return C, V_ori
    

    def get_C_repair(self, pred_y):
        pred_y = pred_y.to(self.device)
        if self.property == 'p2':
            # all other class - class0 >= 0 =====> sat! Using lower bonuds of (f_others - f_0) to check this
            C = torch.zeros(size=(self.unsat_data_num, 4, self.n_classes), device=self.device)
            for i, cls in enumerate(self.constraint):
                C[:, i, 0] = -1.0
                C[:, i, cls] = 1.0
        elif self.property == 'p7':
            # exist j in {0,1,2}, such that (f_3 - f_j >= 0) or (f_4 - f_j >= 0)
            C = torch.zeros(size=(self.unsat_data_num, 6, self.n_classes), device=self.device)
            idx = 0
            for pos_cls in self.constraint:
                for neg_cls in [l for l in range(self.n_classes) if l not in self.constraint]:
                    C[:, idx, pos_cls] = 1.0
                    C[:, idx, neg_cls] = -1.0
                    idx += 1
        elif self.property == 'p8':
            # forall j in {2,3,4}, such that (f_j - f_0 >= 0) or (f_j - f_1 >= 0)
            C = torch.zeros(size=(self.unsat_data_num, 6, self.n_classes), device=self.device)
            idx = 0
            for pos_cls in self.constraint:
                for neg_cls in [l for l in range(self.n_classes) if l not in self.constraint]:
                    C[:, idx, pos_cls] = 1.0
                    C[:, idx, neg_cls] = -1.0
                    idx += 1
        return C
    
    
    def compute_max_effect(self, round, check_layer, model, analyze_neuron_num, N, interval_num):
        print('Now we start analyze the model, compute max effect for every neuron.')
        self.buggy_model.only_logits = False
        all_neuron_effect = torch.zeros((analyze_neuron_num, self.unsat_data_num + 1)).to(self.device)
        all_neuron_eps_effect = torch.zeros((analyze_neuron_num, self.unsat_data_num, interval_num, 2)).to(self.device)
        t = time.time()

        if len(self.unsat_data) == 0:
            return 0, 0

        all_output = self.buggy_model(self.unsat_data)
        logits = all_output[-1]
        true_input = all_output[check_layer]
 
        lirpa_model = BoundedModule(model, torch.empty_like(true_input), device=true_input.device)
        neuron_num = len(true_input[0])

        # shape = neuron_num * interval_num * 2
        eps_record = torch.zeros((analyze_neuron_num, interval_num, 2)).to(self.device)

        st = 0
        end = 10
        steps = (end - st) / interval_num
        intervals = torch.stack([
            torch.linspace(st, end - steps, interval_num),
            torch.linspace(steps + st, end, interval_num)
        ], dim=1)  # Shape: [interval_num, 2]
        eps_record = intervals.unsqueeze(0).expand(analyze_neuron_num, -1, -1).to(self.device)


        # original V
        ori = torch.zeros((self.unsat_data_num)).to(self.device)
        ori = all_output[-1][:, 0]
        C, ori = self.get_C_loc(logits)

        for neuron in tqdm(range(neuron_num)):
            eps = eps_record[neuron]
            true_input_L = true_input.detach().clone()
            true_input_U = true_input.detach().clone()
            # print(eps)
            for interval in range(len(eps)):

                true_input_L[:, neuron] = eps[interval][0]
                true_input_U[:, neuron] = eps[interval][1]
                ptb = PerturbationLpNorm(x_L=true_input_L, x_U=true_input_U)
                true_input = BoundedTensor(true_input, ptb)
                        
                required_A = defaultdict(set)
                required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
                            
                lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=self.approximate_method.split()[0], return_A=True,
                                                                needed_A_dict=required_A, C=C)
                l_A, l_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                                        A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']

                # l_A shape:  self.incor_data_num * |C| * neurons, l_bias shape:  self.incor_data_num * |C|              
                # interval_l_bound_L, interval_l_bound_U shape ---- [self.incor_data_num]
                interval_l_bound_L = torch.sum(true_input_L * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                interval_l_bound_U = torch.sum(true_input_U * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                # shape = neuron_num * self.incor_data_num * interval_num * 2
                all_neuron_eps_effect[neuron, :, interval, 0] = (interval_l_bound_L - ori).detach().clone()
                all_neuron_eps_effect[neuron, :, interval, 1] = (interval_l_bound_U - ori).detach().clone()

            max_vals = all_neuron_eps_effect[neuron].max(dim=-1).values.max(dim=-1).values
            all_neuron_effect[neuron, :self.unsat_data_num] = torch.clamp(max_vals, min=0)

        max_effect = all_neuron_effect[:, :self.unsat_data_num].max(dim=0, keepdim=True).values
        min_effect = all_neuron_effect[:, :self.unsat_data_num].min(dim=0, keepdim=True).values
        diff = max_effect - min_effect
        mask = diff > 0
        all_neuron_effect[:, :self.unsat_data_num] = torch.where(mask, (all_neuron_effect[:, :self.unsat_data_num] - min_effect) / diff, 0)

        all_neuron_effect[:, self.unsat_data_num] = torch.sum(all_neuron_effect[:, 0: N], dim=1)
        sorted_effect, sorted_index = torch.sort(all_neuron_effect[:, self.unsat_data_num], descending=True)
        all_neuron_effect = all_neuron_effect.tolist()
        all_neuron_effect.sort(key=lambda x:x[-1], reverse=True)

        if check_layer not in self.loc_time:
            self.loc_time[check_layer] = 0
        self.loc_time[check_layer] += time.time() - t
        # print(true_label)
        # print(sorted_effect, sorted_index)

        repair_neuron = sorted_index[0]
        print('repair_neuron', sorted_index)
        print('repair_neuron', sorted_effect)  
        print('eps', eps_record[repair_neuron])   

        vio = self.local_repair(check_layer=check_layer, model=model, repair_neuron=repair_neuron, 
                                violation_bounds=all_neuron_eps_effect[repair_neuron], eps_record=eps_record[repair_neuron])
        print('After repair {} '.format(repair_neuron), 'down VR : ', vio[0], ' counter VR : ', vio[1])
        # return sorted_effect, sorted_index, all_neuron_effect, all_neuron_eps_effect, eps_record
        return vio[0], vio[1]


    def local_repair(self, check_layer, model, repair_neuron, violation_bounds, eps_record):
        """
        violation_bounds (Vp*): if there is no ideal interval, use these bounds to determine a single point as an ideal interval
        """
        print('Now we start to repair the model locally')

        t = time.time()
        inputs = torch.cat([self.unsat_data, self.sat_data], dim=0)
        all_output = self.buggy_model(inputs)
        pred_y = torch.min(all_output[-1].cpu(), 1)[1]
        pre_layer_value = all_output[check_layer - 1]
        check_layer_value = all_output[check_layer]
        unsat_h = check_layer_value[0: self.unsat_data_num]
        # before repair
        check_neuron_value = check_layer_value[:, repair_neuron]
        in_weight_num = len(pre_layer_value[0])

        # ideal interval
        target = torch.empty([self.sat_data_num + self.unsat_data_num, 2], requires_grad=False).to(self.device)
        C = self.get_C_repair(pred_y[0: self.unsat_data_num])

        lirpa_model = BoundedModule(model, torch.empty_like(unsat_h), device=unsat_h.device)  
        h_L = unsat_h.detach().clone()
        h_U = unsat_h.detach().clone()

        # recodre all intervals are: sat or not
        ideal = torch.zeros((self.unsat_data_num, len(eps_record)), dtype=torch.bool).to(self.device)
        for interval in range(len(eps_record)):

            h_L[:, repair_neuron] = eps_record[interval][0]
            h_U[:, repair_neuron] = eps_record[interval][1]
            ptb = PerturbationLpNorm(x_L=h_L, x_U=h_U)
            true_input = BoundedTensor(unsat_h, ptb)
                    
            required_A = defaultdict(set)
            required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])
                        
            lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=self.approximate_method.split()[0], return_A=True,
                                                            needed_A_dict=required_A, C=C)
            l_A, l_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                        A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']

            # lb: N * |C|, e.g., N * 4
            if self.conj:
                # constraints are conjunctive
                ideal[:, interval] = torch.min(lb, dim=1)[0] > 0
            else:
                #p7: N * 6 --> N * 3 * 2: (f3>f0 and f4>f0) or (f3>f1 and f4>f1) or (f3>f2 and f4>f2)
                #p8: N * 6 --> N * 2 * 3: (f2>f0 and f3>f0 and f4>f0) or (f2>f1 and f3>f1 and f4>f1)
                lb = lb.view(lb.shape[0], -1, len(self.constraint))
                # N * 3 or N * 2
                after_conj = torch.min(lb, dim=2)[0] > 0
                ideal[:, interval] = torch.max(after_conj, dim=1)[0] > 0


        has_ideal = ideal.any(dim=1)

        # Consider data samples that have at least one interval satisfying C1
        if has_ideal.any():
            # shape of d0 - [N, k]
            print(eps_record[..., 0].shape, unsat_h.unsqueeze(1).shape)
            d0 = torch.abs(eps_record[..., 0] - unsat_h[:, repair_neuron].unsqueeze(1))
            d1 = torch.abs(eps_record[..., 1] - unsat_h[:, repair_neuron].unsqueeze(1))
            # shape of min_d0_d1 - [N, k]
            min_d0_d1 = torch.minimum(d0, d1)
            print(ideal.shape, min_d0_d1.shape)
            min_d0_d1[~ideal] = float('inf')
            _, opt_index = torch.min(min_d0_d1, dim=1)
            
            selected_points = eps_record[opt_index]
            print(selected_points.shape)
            target[:self.unsat_data_num][has_ideal] = selected_points[has_ideal]

        if not has_ideal.all():
            m, _ = torch.max(violation_bounds, dim=2)
            _, opt_index = torch.max(m, dim=1)

            selected_points = eps_record[opt_index]
            target[:self.unsat_data_num][~has_ideal] = selected_points[~has_ideal]

        if not has_ideal.all():
            subset = violation_bounds[~has_ideal]  # [m, K, 2]
            m_vals, m_idx = torch.max(subset, dim=2)  # [m, K], [m, K]
            best_vals, opt_index = torch.max(m_vals, dim=1)  # [m], [m]
            index_side = m_idx[torch.arange(m_idx.size(0), device=self.device), opt_index]  # [m]
            eps_chosen = eps_record[opt_index]  # [m, 2]
            ideal_endpoint = eps_chosen[torch.arange(eps_chosen.size(0), device=self.device), index_side]  # [m]
            ideal_endpoint = ideal_endpoint.unsqueeze(1).expand(-1, 2) # [m, 2]
            target[:self.unsat_data_num][~has_ideal] = ideal_endpoint


        target[self.unsat_data_num:] = check_neuron_value[self.unsat_data_num:].unsqueeze(1).detach()

        d = len(self.buggy_model.state_dict().keys()) - len(model.state_dict().keys())

        para_name = list(self.buggy_model.state_dict().keys())
        pre_check_wegiht = self.buggy_model.state_dict()[para_name[d - 2]]
        pre_check_bias = self.buggy_model.state_dict()[para_name[d - 1]]
        
        post_check_wegiht = self.buggy_model.state_dict()[para_name[d]]
        post_check_bias = self.buggy_model.state_dict()[para_name[d + 1]]
        # print('-', post_check_wegiht.shape, post_check_bias.shape)
        # weight size: check layer * (check layer - 1)
        # print(pre_check_wegiht, pre_check_wegiht.shape)
        # print(pre_check_bias, pre_check_bias.shape)
        
        mini_input = torch.empty([self.sat_data_num + self.unsat_data_num, in_weight_num], requires_grad=False).to(self.device)
        mini_input = pre_layer_value.detach()
        
        flags = torch.zeros(target.shape[0], 1).to(self.device)
        flags[:self.unsat_data_num] = 1.0
        
        mini_data = TensorDataset(mini_input, target.detach(), flags)
        mini_loader = DataLoader(dataset=mini_data, batch_size=self.BATCH_SIZE, shuffle=True)

        check_neuron_in_weight = pre_check_wegiht[repair_neuron]
        check_neuron_bias = pre_check_bias[repair_neuron]
        # print(check_neuron_in_weight.shape, check_neuron_bias)
        
        mini_nn = nn.Sequential(
            nn.Linear(in_weight_num, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )
        para = {}
        para['0.weight'] = check_neuron_in_weight.view(1, in_weight_num)
        para['0.bias'] = check_neuron_bias.view(1)
        para['2.weight'] = torch.tensor(1).view(1, 1)
        para['2.bias'] = torch.tensor(0).view(1)
        mini_nn.load_state_dict(para)
        mini_nn.to(self.device)
        mini_nn.train()
        optim = torch.optim.Adam(mini_nn.parameters(), lr=0.01)

        for epoch in range(self.local_epoch):
            for step, (x, y, flag) in enumerate(mini_loader):
                output = mini_nn(x).squeeze(1)
                is_incor = flag.view(-1).bool()
                is_cor = ~is_incor
                loss = torch.tensor(0.0).to(self.device)
                
                if is_incor.any():
                    out_inc = output[is_incor]
                    y_inc = y[is_incor]
                    l0 = out_inc - y_inc[:, 0]
                    l1 = out_inc - y_inc[:, 1]
                    loss_repair = torch.where((l0 > 0) & (l1 > 0), l1 ** 2, torch.zeros_like(l1)) + \
                                  torch.where((l0 < 0) & (l1 < 0), l0 ** 2, torch.zeros_like(l0))
                    loss += loss_repair.mean()

                if is_cor.any():
                    out_cor = output[is_cor]
                    y_cor = y[is_cor]
                    loss_fidelity = (out_cor - y_cor[:, 0]) ** 2
                    loss += loss_fidelity.mean()                

                if loss == 0:
                    continue
                optim.zero_grad()  
                loss.backward()  
                optim.step()
        
        repaired_para = {}
        for key in self.buggy_model.state_dict().keys():
            repaired_para[key] = self.buggy_model.state_dict()[key]
        b = mini_nn.state_dict()['2.bias']
        w = mini_nn.state_dict()['2.weight'][0]
        repaired_para[para_name[d - 2]][repair_neuron] = mini_nn.state_dict()['0.weight']
        repaired_para[para_name[d - 1]][repair_neuron] = mini_nn.state_dict()['0.bias']
        repaired_para[para_name[d + 1]] += b * post_check_wegiht[:, repair_neuron]
        # repaired_para[para_name[d]] *= w
        repaired_para[para_name[d]][:, repair_neuron] *= w
            
        if check_layer not in self.repair_time:
            self.repair_time[check_layer] = 0
        self.repair_time[check_layer] += time.time() - t
        self.buggy_model.load_state_dict(repaired_para)
        print('-' * 20, 'One repair round end, using data (for repair) to test!')
        normal_vio = self.Test_VR(self.buggy_model, self.cor_data)
        mis_vio = self.Test_VR(self.buggy_model, self.vio_data)      
        return normal_vio, mis_vio

   