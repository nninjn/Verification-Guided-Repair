import os
import time
import numpy as np
import torch
import torch.nn as nn
from utils.prepare import prepare_dataloaders

from collections import defaultdict
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
import pyswarms as ps
from tqdm import tqdm

class Repair():
    def __init__(self, BATCH_SIZE, n_classes, dataset, target, 
                 clean_model, poisoned_model, clean_test_data, poisoned_test_data,\
                 approximate_method, N, N_clean, local_epoch, device) -> None:

        self.BATCH_SIZE = BATCH_SIZE
        self.target = target
        self.clean_model = clean_model
        self.poisoned_model = poisoned_model
        self.device = device
        self.poisoned_model.eval()
        self.poisoned_model.to(self.device)

        self.clean_test_data = clean_test_data
        self.poisoned_test_data = poisoned_test_data
        self.n_classes = n_classes

        self.approximate_method = approximate_method

        self.way = []

        self.N = N
        self.N_clean = N_clean
        self.local_epoch = local_epoch

        self.repair_time = {}
        self.loc_time = {}
        self.imagenet = True if 'imagenet' in dataset else False

        self.test_acc_dataset, self.test_sr_dataset, self.repair_clean_dataset, self.poi_data_for_repair,  \
        self.test_acc_loader, self.test_sr_loader, self.clean_data_for_repair_loader, self.poi_data_for_repair_loader = \
                prepare_dataloaders(clean_test_data=self.clean_test_data, poisoned_test_data=self.poisoned_test_data,
                                    BATCH_SIZE=BATCH_SIZE, N_clean=self.N_clean, N_poi=self.N, imagenet=self.imagenet)

        self.cor_data = []
        self.incor_data = []
        self.cor_data_num = len(self.cor_data)
        self.incor_data_num = len(self.incor_data)

        self.classifi = 0
        
    def Test_acc(self, nn_test, loader, model_type=None):
        # print('Test Start! Test Type = Clean data Accuracy!')
        nn_test.eval()
        nn_test = nn_test.to(self.device)
        accuracy = 0
        for step, (x, y) in enumerate(loader):
            train_output = nn_test(x.to(self.device))[-1]
            pred_y = torch.max(train_output.cpu(), 1)[1].numpy()
            label_y = y.cpu().numpy()
            accuracy += (pred_y == label_y).sum()
        print(accuracy, len(loader.dataset))
        print('-' * 20, 'Acc = ', accuracy / len(loader.dataset) * 100)
        return accuracy / len(loader.dataset)


    def Test_SR(self, nn_test, loader, model_type=None):
        # print('Test Start! Test Type = Poisoned data SR!', ' Model Type = ', model_type, 'target:', self.target)
        nn_test.eval()
        nn_test = nn_test.to(self.device)
        success = 0
        nocount = 0
        for step, (x, y) in enumerate(loader):
            train_output = nn_test(x.to(self.device))[-1]
            pred_y = torch.max(train_output.cpu(), 1)[1].numpy()
            label_y = y.cpu().numpy()
            success += ((pred_y == self.target) & (label_y != self.target)).sum()
            nocount += (label_y == self.target * np.ones(pred_y.shape)).sum()
        print(nocount, success, len(loader.dataset))
        sr = success / (len(loader.dataset) - nocount)
        print('-' * 20, 'SR = ', sr * 100)
        # return sr, len(loader.dataset) - nocount - success
        return sr


    @torch.no_grad()
    def repair_data_classification(self):
        self.poisoned_model.eval()

        def get_preds(loader):
            all_preds, all_labels = [], []
            for x, y in loader:
                out = self.poisoned_model(x.to(self.device))[-1]
                all_preds.append(out.argmax(dim=1).cpu())
                all_labels.append(y)
            return torch.cat(all_preds), torch.cat(all_labels)

        preds_c, labels_c = get_preds(self.clean_data_for_repair_loader)
        cor_idx = (preds_c == labels_c).nonzero(as_tuple=True)[0]
        self.cor_data = [self.repair_clean_dataset[i] for i in cor_idx]
        preds_p, labels_p = get_preds(self.poi_data_for_repair_loader)
        incor_mask = (preds_p == self.target) & (labels_p != self.target)
        incor_idx = incor_mask.nonzero(as_tuple=True)[0]
        
        batch = 50 if self.imagenet else 128
        self.incor_data = [self.poi_data_for_repair[i] for i in incor_idx[:batch]]
        self.cor_data_num, self.incor_data_num = len(self.cor_data), len(self.incor_data)
        # print(f"Correct: {self.cor_data_num}, Incorrect: {self.incor_data_num}")


    def compute_max_effect(self, round, check_layer, model, interval_num):
        with torch.no_grad():
            print(f'Fault localization round {round}.')
            t = time.time()
            if len(self.incor_data) == 0:
                return 0, 0

            image = torch.stack([x for x, _ in self.incor_data]).to(self.device)

            true_label = [y for _, y in self.incor_data]
            all_output = self.poisoned_model(image)
            pred_y = torch.max(all_output[-1].cpu(), 1)[1]
            true_input = all_output[check_layer].to(self.device)

            lirpa_model = BoundedModule(model, torch.empty_like(true_input), device=true_input.device)

            neuron_num = len(true_input[0])
            all_neuron_effect = torch.zeros((neuron_num, self.incor_data_num + 1)).to(self.device)
            # shape = neuron_num * N * interval_num * 2
            
            all_neuron_eps_effect = torch.zeros((neuron_num, self.incor_data_num, interval_num, 2)).to(self.device)
            if self.poisoned_model.act is nn.Sigmoid:
                st = -1
                end = 1
                steps = (end - st) / interval_num
                intervals = torch.stack([
                    torch.linspace(st, end - steps, interval_num),
                    torch.linspace(steps + st, end, interval_num)
                ], dim=1)  # Shape: [interval_num, 2]
            else:
                st, end = 0, 10
                starts = torch.arange(interval_num, device=self.device).float() * (end - st) + st
                ends = starts + (end - st) 
                intervals = torch.stack([starts, ends], dim=1)

            eps_record = torch.zeros((neuron_num, interval_num, 2)).to(self.device)
            eps_record = intervals.unsqueeze(0).expand(neuron_num, -1, -1).to(self.device)

            C = torch.zeros(size=(self.incor_data_num, 1, self.n_classes), device=true_input.device)
            #  true label - target label
            for i in range(self.incor_data_num):
                C[i][0][true_label[i]] = 1.0
                C[i][0][pred_y[i]] = -1.0   
            ori = torch.zeros((self.incor_data_num)).to(self.device)
            for i in range(self.incor_data_num):
                ori[i] = all_output[-1][i][true_label[i]] - all_output[-1][i][pred_y[i]]

            for neuron in tqdm(range(neuron_num)):              
                eps = eps_record[neuron]
                true_input_L = true_input.detach().clone()
                true_input_U = true_input.detach().clone()
                # print(eps)
                for interval in range(len(eps)):
                    true_input_L[:, neuron] = eps[interval][0]
                    true_input_U[:, neuron] = eps[interval][1]
                    ptb = PerturbationLpNorm(x_L=true_input_L.detach().clone(), x_U=true_input_U.detach().clone())
                    true_input = BoundedTensor(true_input.detach().clone(), ptb)
                    required_A = defaultdict(set)
                    required_A[lirpa_model.output_name[0]].add(lirpa_model.input_name[0])

                    lb, ub, A_dict = lirpa_model.compute_bounds(x=(true_input,), method=self.approximate_method, return_A=True,
                                                                    needed_A_dict=required_A, C=C)
                    l_A, l_bias = A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lA'], \
                                            A_dict[lirpa_model.output_name[0]][lirpa_model.input_name[0]]['lbias']
                    # l_A shape:  self.incor_data_num * 1 * neurons
                    # l_bias shape:  self.incor_data_num * 1
                    # interval_l_bound_L interval_l_bound_U shape ---- [self.incor_data_num]
                    interval_l_bound_L = torch.sum(true_input_L * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                    interval_l_bound_U = torch.sum(true_input_U * l_A[:, 0, :], dim=1) + l_bias[:, 0]
                    # shape = neuron_num * self.incor_data_num * interval_num * 2
                    all_neuron_eps_effect[neuron, :, interval, 0] = (interval_l_bound_L - ori).detach().clone()
                    all_neuron_eps_effect[neuron, :, interval, 1] = (interval_l_bound_U - ori).detach().clone()

                max_vals = all_neuron_eps_effect[neuron].max(dim=-1).values.max(dim=-1).values
                all_neuron_effect[neuron, :self.incor_data_num] = torch.clamp(max_vals, min=0)

            max_effect = all_neuron_effect[:, :self.incor_data_num].max(dim=0, keepdim=True).values
            min_effect = all_neuron_effect[:, :self.incor_data_num].min(dim=0, keepdim=True).values
            diff = max_effect - min_effect
            mask = diff > 0
            all_neuron_effect[:, :self.incor_data_num] = torch.where(mask, (all_neuron_effect[:, :self.incor_data_num] - min_effect) / diff, 0)
            all_neuron_effect[:, self.incor_data_num] = torch.sum(all_neuron_effect[:, 0: self.incor_data_num], dim=1)

            sorted_effect, sorted_index = torch.sort(all_neuron_effect[:, self.incor_data_num], descending=True)
            all_neuron_effect = all_neuron_effect.tolist()
            all_neuron_effect.sort(key=lambda x:x[-1], reverse=True)
            if check_layer not in self.loc_time:
                self.loc_time[check_layer] = 0
            self.loc_time[check_layer] += time.time() - t
        # print(sorted_index)
        # print(sorted_effect)
        # print(sorted_index[0])
        # print(eps_record[sorted_index[0]])

        repair_neuron = sorted_index[0]
        self.way.append(repair_neuron)
        self.new_repair(check_layer=check_layer, model=model, repair_interval=all_neuron_eps_effect[repair_neuron],\
                                analyze_neuron=repair_neuron, ori=ori, eps_record=eps_record[repair_neuron])
        acc = self.Test_acc(self.poisoned_model, self.clean_data_for_repair_loader, 'Poisoned Model')
        sr = self.Test_SR(self.poisoned_model, self.poi_data_for_repair_loader, 'Poisoned Model')      

        print('After repair {} '.format(repair_neuron), 'the model acc : ', acc, ' sr : ', sr)
        return acc, sr


    def new_repair(self, check_layer, model, repair_interval, analyze_neuron, ori, eps_record):
        print('Now we start to repair the model locally')
        start = time.time()
        image = [x for x, _ in self.incor_data] + [x for x, _ in self.cor_data]
        image = torch.stack(image).to(self.device)
        # print(image.shape, image.shape[0])
       
        with torch.no_grad():
            all_output = self.poisoned_model(image)
        pre_layer_value = all_output[check_layer - 1]
        check_layer_value = all_output[check_layer]
        check_neuron_value = check_layer_value[:, analyze_neuron]

        # mini-nn label
        target = torch.zeros([self.incor_data_num + self.cor_data_num, 2], requires_grad=False).to(self.device)
        repair_interval = repair_interval.to(self.device)  # shape [N, K, 2]
        ori = ori.to(self.device)  # shape [N]
        eps_record = eps_record.to(self.device)  # shape [K, 2]

        # check ideal interval
        cond = (repair_interval[:, :, 0] + ori.unsqueeze(1) > 0) & \
               (repair_interval[:, :, 1] + ori.unsqueeze(1) > 0)
        has_point_set = cond.any(dim=1)  # [N]
        not_entirely_idx = ~has_point_set

        # case 1: consider the inputs that donot have ideal interval
        if not_entirely_idx.any():
            subset = repair_interval[not_entirely_idx]  # [m, K, 2]
            m_vals, m_idx = torch.max(subset, dim=2)  # [m, K]
            best_vals, best_j = torch.max(m_vals, dim=1)  # [m]
            index_side = m_idx[torch.arange(m_idx.size(0)), best_j]  # [m]
            eps_chosen = eps_record[best_j]  # [m, 2]
            vals = eps_chosen[torch.arange(eps_chosen.size(0), device=self.device), index_side]  # [m]
            vals = vals.unsqueeze(1).expand(-1, 2)
            target[torch.arange(self.incor_data_num, device=self.device)[not_entirely_idx]] = vals

        # case 2: consider the inputs that have ideal interval
        if has_point_set.any():
            h_value = check_neuron_value[:self.incor_data_num][has_point_set].unsqueeze(1)  # [m', 1]
            cond_subset = cond[has_point_set]  # [m', K]
            diff0 = torch.abs(eps_record[:, 0].unsqueeze(0) - h_value)  # [m', K]
            diff1 = torch.abs(eps_record[:, 1].unsqueeze(0) - h_value)  # [m', K]
            d_min = torch.minimum(diff0, diff1)
            d_min_masked = torch.where(cond_subset, d_min, torch.full_like(d_min, 1e9))
            # index of ideal interval
            opt_idx = torch.argmin(d_min_masked, dim=1)  # [m']
            target[:self.incor_data_num][has_point_set] = eps_record[opt_idx]

        target[self.incor_data_num:, 0] = check_neuron_value[self.incor_data_num:]
        target[self.incor_data_num:, 1] = check_neuron_value[self.incor_data_num:]
        target = target.detach()

        # print(self.poisoned_model.state_dict().keys())
        # print(model.state_dict().keys())
        d = len(self.poisoned_model.state_dict().keys()) - len(model.state_dict().keys())

        para_name = list(self.poisoned_model.state_dict().keys())
        pre_check_wegiht = self.poisoned_model.state_dict()[para_name[d - 2]]
        pre_check_bias = self.poisoned_model.state_dict()[para_name[d - 1]]
        
        # weight size: check layer * (check layer - 1)
        post_check_wegiht = self.poisoned_model.state_dict()[para_name[d]]
        post_check_bias = self.poisoned_model.state_dict()[para_name[d + 1]]
        # print('-', post_check_wegiht.shape, post_check_bias.shape)

        mini_input = pre_layer_value.detach()

        flags = torch.zeros(target.shape[0], 1).to(self.device)
        flags[:self.incor_data_num] = 1.0
        
        mini_data = TensorDataset(mini_input, target, flags)
        mini_loader = DataLoader(dataset=mini_data, batch_size=self.BATCH_SIZE, shuffle=True)
        print(pre_check_wegiht.shape, pre_check_bias.shape)
        check_neuron_in_weight = pre_check_wegiht[analyze_neuron]
        check_neuron_bias = pre_check_bias[analyze_neuron]
        in_weight_num = len(pre_layer_value[0])
        # print(check_neuron_in_weight.shape, check_neuron_bias)
        new_neuron = nn.Sequential(
            nn.Linear(in_weight_num, 1),
            self.poisoned_model.act(),
            nn.Linear(1, 1)
        )
        para = {}
        para['0.weight'] = check_neuron_in_weight.view(1, in_weight_num)
        para['0.bias'] = check_neuron_bias.view(1)
        para['2.weight'] = torch.tensor(1).view(1, 1)
        para['2.bias'] = torch.tensor(0).view(1)
        new_neuron.load_state_dict(para)
        new_neuron.to(self.device)
        new_neuron.train()
        optim = torch.optim.Adam(new_neuron.parameters(), lr=0.01)
        if self.imagenet:
            optim = torch.optim.Adam(new_neuron.parameters(), lr=0.001)

        for epoch in range(self.local_epoch):
            for step, (x, y, flag) in enumerate(mini_loader):
                output = new_neuron(x).squeeze(1)
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
        for key in self.poisoned_model.state_dict().keys():
            repaired_para[key] = self.poisoned_model.state_dict()[key]
        b = new_neuron.state_dict()['2.bias']
        w = new_neuron.state_dict()['2.weight'][0]
        repaired_para[para_name[d - 2]][analyze_neuron] = new_neuron.state_dict()['0.weight']
        # print('pre weight shape:', repaired_para[para_name[d - 2]][analyze_neuron].shape, mini_nn.state_dict()['0.weight'].shape)
        repaired_para[para_name[d - 1]][analyze_neuron] = new_neuron.state_dict()['0.bias']
        # print('pre bias shape:', repaired_para[para_name[d - 1]][analyze_neuron].shape, mini_nn.state_dict()['0.bias'].shape)
        repaired_para[para_name[d + 1]] += b * post_check_wegiht[:, analyze_neuron]
        repaired_para[para_name[d]][:, analyze_neuron] *= w
        
        if check_layer not in self.repair_time:
            self.repair_time[check_layer] = 0
        self.repair_time[check_layer] += time.time() - start
        self.poisoned_model.load_state_dict(repaired_para)

   
