from collections import defaultdict
import torch
import torch.nn as nn
import numpy as np
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
import time
from dataprocess import bank_data
from utils.utils import Data, test_acc
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset

class Repair():

    def __init__(self, BATCH_SIZE, model, nor_data, un_data, approximate_method, interval_times, local_epoch, device) -> None:

        self.BATCH_SIZE = BATCH_SIZE
        self.model = model
        self.model.eval()
        self.device = device

        self.normal_data = torch.from_numpy(nor_data).float().to(self.device)
        self.unfair_data = torch.from_numpy(un_data).float().to(self.device)
        
        self.appr_method = approximate_method
        self.local_epoch = local_epoch
        self.alpha = 0.1
        self.repair_n = 20
        self.pso_repair_way = []

        self.interval_num = interval_times

        self.incor_data = []
        self.cor_data_num = len(self.normal_data)
        self.incor_data_num = len(self.unfair_data)
        
        self.loc_time = 0
        self.repair_time = 0


    def Test_VR(self, model, loader):
        PAIR_SIZE = 2
        model.eval().to(self.device)
        model_dtype = next(model.parameters()).dtype
        mismatch_count = 0
        total_pairs = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device=self.device, dtype=model_dtype)
                if batch.dim() == 0 or batch.size(0) == 0:
                    print("Warning: Batch is empty. Skipping this batch.")
                    continue

                batch_size = batch.size(0) // PAIR_SIZE
                if batch_size == 0:
                    print("Warning: Batch size is zero. Skipping this batch.")
                    continue

                paired_batch = batch.view(batch_size, PAIR_SIZE, -1)
                outputs = model(paired_batch.flatten(0, 1))
                preds = (outputs > 0.5).int().view(batch_size, PAIR_SIZE)
                
                batch_mismatch = torch.ne(preds[:, 0], preds[:, 1]).sum().item()
                mismatch_count += batch_mismatch
                total_pairs += batch_size
        return mismatch_count / total_pairs if total_pairs else 0.0


    def test_fairness(self, model, loaders):
        total_pairs = 0
        total_violation_pairs = 0.0
        results = {}
        for idx, loader in enumerate(loaders, 1):
            current_pairs = loader.size(0)
            violation_rate = self.Test_VR(model, loader)
            results[f"Loader {idx}"] = {
                "violation_rate": violation_rate,
                "total_samples": current_pairs,
                "total_pairs": current_pairs
            }
            total_pairs += current_pairs
            total_violation_pairs += violation_rate * current_pairs
        weighted_avg = total_violation_pairs / total_pairs if total_pairs > 0 else 0.0
        return weighted_avg
    

    def total_test(self, nor_feature_tensor, nor_label, unf_loader, nor_num, unf_num, model, verbose=False):
        unf_results = self.test_fairness(model=model, loaders=unf_loader)
        # nor_results = test_acc(model, nor_feature_tensor, torch.tensor(nor_label, dtype=torch.long))
        nor_results = test_acc(model, nor_feature_tensor, torch.tensor(nor_label))

        if verbose:
            print("=" * 50)
            print("[ Normal dataset ]")
            print(f"│   total_num: {nor_num}")
            print(f"│   acc: {nor_results}")
            print("└─" + "─" * 50)

            print("[ Unfair Train ]")
            print(f"│ total_num: {unf_num:4d}")
            print(f"│ unfair_rate: {unf_results:.4f}")
            print("└─" + "─" * 50)
        best = nor_num * nor_results + unf_num * (1 - unf_results)
        return best, nor_results, unf_results
    

    @torch.no_grad()
    def repair_data_classification(self):
        self.model.eval()
        x1 = self.unfair_data[:, 0, :]
        x2 = self.unfair_data[:, 1, :]
        out1 = self.model(x1.to(self.device))
        out2 = self.model(x2.to(self.device))
        preds_1 = (out1 > 0.5).long().squeeze()
        preds_2 = (out2 > 0.5).long().squeeze()
        unfair_mask = (preds_1 != preds_2)
        unfair_pairs = self.unfair_data[unfair_mask]

        self.incor_data = unfair_pairs.detach().clone()
        self.incor_data_num = len(self.incor_data)
        
        # print(f"We now have {unfair_mask.sum().item()} unfair pairs.")


    def analyze_neuron_action(self, neuron_num, cur_h1, cur_h2, dynamic=True):
        if(dynamic == True):
            neuron_act_bound = torch.zeros(neuron_num, self.incor_data_num, 2)
            for i in range(neuron_num):
                for j in range(self.incor_data_num):
                    neuron_act_bound[i][j][0] = min(cur_h1[j][i],cur_h2[j][i])
                    neuron_act_bound[i][j][1] = max(cur_h1[j][i],cur_h2[j][i])
                    
                    # neuron_act_bound[i][j][0] = 0
                    # neuron_act_bound[i][j][1] = 10
        else:
            check_value = torch.cat((cur_h1, cur_h2), dim=0)
            neuron_act_bound = torch.stack([
                check_value.min(dim=0).values,
                check_value.max(dim=0).values
            ], dim=1)  # [neurons, 2]
        return neuron_act_bound
    

    def split_intervals(self, neuron_act_bound, k):
        lower = neuron_act_bound[..., 0]#shape: [16, 200]
        upper = neuron_act_bound[..., 1]
        
        steps = torch.linspace(0, 1, steps=k+1, device=neuron_act_bound.device)# shape[16, 200, k+1]
        split_points = lower.unsqueeze(-1) + (upper - lower).unsqueeze(-1) * steps
        
        eps_record = torch.stack([   #shape[16, 200, k, 2]
            split_points[..., :-1],
            split_points[..., 1:]
        ], dim=-1)
        return eps_record


    def compute_max_effect(self, check_layer, model, neuron_num, nor_feature):

        all_input_effect = torch.zeros((neuron_num)).to(self.device)
        all_neuron_effect = torch.zeros((neuron_num, self.incor_data_num)).to(self.device)
        all_endpoints_1 = torch.zeros((neuron_num, self.incor_data_num, self.interval_num, 4)).to(self.device)
        all_endpoints_2 = torch.zeros((neuron_num, self.incor_data_num, self.interval_num, 4)).to(self.device)
        t = time.time()

        if self.incor_data_num == 0:
            return 0, 0
        
        data_nor = torch.from_numpy(nor_feature).float().to(self.device)
        data_x1 = self.incor_data[:, 0, :] #+
        data_x2 = self.incor_data[:, 1, :] #-

        all_output_1 = self.model(data_x1, return_all_outputs=True)
        all_output_2 = self.model(data_x2, return_all_outputs=True)
        cur_h1 = all_output_1[check_layer]#shape: num * neuron_num
        cur_h2 = all_output_2[check_layer]
        last_h1 = all_output_1[-1]
        last_h2 = all_output_2[-1]
        result = (last_h1 > -last_h2).int()
        result_squeezed = result.squeeze(dim=1)

        neuron_act_bound = self.analyze_neuron_action(neuron_num, cur_h1, cur_h2)
        eps_record = self.split_intervals(neuron_act_bound, self.interval_num)

        lirpa_model_1 = BoundedModule(model, cur_h1, device=cur_h1.device)
        lirpa_model_2 = BoundedModule(model, cur_h2, device=cur_h2.device)
        base_h1 = cur_h1.detach().clone() #incor_data_num * neuron_num
        base_h2 = cur_h2.detach().clone()
        eps_record = eps_record.to(base_h1.device)

        required_A_1 = defaultdict(set)
        required_A_1[lirpa_model_1.output_name[0]].add(lirpa_model_1.input_name[0])
        required_A_2 = defaultdict(set)
        required_A_2[lirpa_model_2.output_name[0]].add(lirpa_model_2.input_name[0])
        
        for neuron in range(neuron_num):
            eps = eps_record[neuron]  # (incor_data_num, self.interval_num, 2)
            
            batch_size = self.incor_data_num * self.interval_num
            
            # self.interval_num * incor_data_num * neuron_num
            h1_expanded = base_h1.unsqueeze(1).expand(-1, self.interval_num, -1).reshape(batch_size, -1)
            eps_L = eps[:, :, 0].reshape(-1)
            eps_U = eps[:, :, 1].reshape(-1)
            h1_L = h1_expanded.clone()
            h1_U = h1_expanded.clone()
            h1_L[:, neuron] = eps_L
            h1_U[:, neuron] = eps_U
            
            h2_expanded = base_h2.unsqueeze(1).expand(-1, self.interval_num, -1).reshape(batch_size, -1)
            h2_L = h2_expanded.clone()
            h2_U = h2_expanded.clone()
            h2_L[:, neuron] = eps_L
            h2_U[:, neuron] = eps_U
            
            with torch.no_grad():
                ptb_1 = PerturbationLpNorm(x_L=h1_L, x_U=h1_U)
                bounded_input_1 = BoundedTensor(h1_expanded, ptb_1)#Kincor_num * neuron_num
                lb_1, ub_1, A_dict_1 = lirpa_model_1.compute_bounds(x=(bounded_input_1,), method=self.appr_method, return_A=True, needed_A_dict=required_A_1)
                lb_1 = lb_1.reshape(self.incor_data_num, self.interval_num).squeeze(-1)
                ub_1 = ub_1.reshape(self.incor_data_num, self.interval_num).squeeze(-1)
                
                ptb_2 = PerturbationLpNorm(x_L=h2_L, x_U=h2_U)
                bounded_input_2 = BoundedTensor(h2_expanded, ptb_2)
                lb_2, ub_2, A_dict_2 = lirpa_model_2.compute_bounds(x=(bounded_input_2,), method=self.appr_method, return_A=True, needed_A_dict=required_A_2)
                lb_2 = lb_2.reshape(self.incor_data_num, self.interval_num).squeeze(-1)
                ub_2 = ub_2.reshape(self.incor_data_num, self.interval_num).squeeze(-1)
            # print(lb_1.shape, ub_1.shape)
            min_ub_1, _ = ub_1.min(dim=1)
            max_lb_2, _ = lb_2.max(dim=1)

            # all_neuron_effect[neuron] = torch.max(-min_ub_1, max_lb_2)
            # all_neuron_effect[neuron] = torch.where(result_squeezed == 1, max_lb_2, -min_ub_1)

            l_A_1 = A_dict_1[lirpa_model_1.output_name[0]][lirpa_model_1.input_name[0]]['lA']
            l_bias_1 = A_dict_1[lirpa_model_1.output_name[0]][lirpa_model_1.input_name[0]]['lbias']
            u_A_1 = A_dict_1[lirpa_model_1.output_name[0]][lirpa_model_1.input_name[0]]['uA']
            u_bias_1 = A_dict_1[lirpa_model_1.output_name[0]][lirpa_model_1.input_name[0]]['ubias']

            input_matrix_1 = torch.stack([h1_L, h1_U, h1_L, h1_U], dim=1)#[self.incor_data_num, self.interval_num, 4]
            weights_1 = torch.stack([l_A_1[:, 0, :], l_A_1[:, 0, :] ,u_A_1[:, 0, :], u_A_1[:, 0, :]], dim=1)
            biases_1 = torch.stack([l_bias_1[:, 0], l_bias_1[:, 0] ,u_bias_1[:, 0], u_bias_1[:, 0]], dim=1)

            effects_1 = torch.einsum('bij,bij->bi', input_matrix_1, weights_1) + biases_1
            all_endpoints_1[neuron] = effects_1.reshape(self.incor_data_num, self.interval_num, 4)
 
            l_A_2 = A_dict_2[lirpa_model_2.output_name[0]][lirpa_model_2.input_name[0]]['lA']
            l_bias_2 = A_dict_2[lirpa_model_2.output_name[0]][lirpa_model_2.input_name[0]]['lbias']
            u_A_2 = A_dict_2[lirpa_model_2.output_name[0]][lirpa_model_2.input_name[0]]['uA']
            u_bias_2 = A_dict_2[lirpa_model_2.output_name[0]][lirpa_model_2.input_name[0]]['ubias']
            
            input_matrix_2 = torch.stack([h2_L, h2_U, h2_L, h2_U], dim=1)
            weights_2 = torch.stack([l_A_2[:, 0, :], l_A_2[:, 0, :], u_A_2[:, 0, :], u_A_2[:, 0, :]], dim=1)
            biases_2 = torch.stack([l_bias_2[:, 0], l_bias_2[:, 0], u_bias_2[:, 0], u_bias_2[:, 0]], dim=1)
            effects_2 = torch.einsum('bij,bij->bi', input_matrix_2, weights_2) + biases_2
            all_endpoints_2[neuron] = effects_2.reshape(self.incor_data_num, self.interval_num, 4)

            max_lb = all_endpoints_2[neuron][:, :, 0:2].max(dim=-1).values.max(dim=-1).values
            min_ub = all_endpoints_1[neuron][:, :, 2:4].min(dim=-1).values.min(dim=-1).values
            all_neuron_effect[neuron] = torch.where(result_squeezed == 1, max_lb, -min_ub)

        max_effect = all_neuron_effect[:, :self.incor_data_num].max(dim=0, keepdim=True).values
        min_effect = all_neuron_effect[:, :self.incor_data_num].min(dim=0, keepdim=True).values
        diff = max_effect - min_effect
        mask = diff > 0
        all_neuron_effect[:, :self.incor_data_num] = torch.where(mask, (all_neuron_effect[:, :self.incor_data_num] - min_effect) / diff, 0)
        
        # NOTE check
        all_input_effect = torch.sum(all_neuron_effect, dim=1)
        _, sorted_index = torch.sort(all_input_effect, descending=True)
        all_neuron_effect = all_neuron_effect.tolist()
        all_neuron_effect.sort(key=lambda x:x[-1], reverse=True)


        self.loc_time += time.time() - t
        repair_neuron = sorted_index[0]

        self.new_repair(check_layer=check_layer, model=model, endpoints_1=all_endpoints_1[repair_neuron],\
            endpoints_2=all_endpoints_2[repair_neuron], analyze_neuron=repair_neuron,\
            eps_record=eps_record[repair_neuron], result=result_squeezed)


    def new_repair(self, check_layer, model, endpoints_1, endpoints_2, analyze_neuron, eps_record, result):

        t = time.time()
        eps_record = eps_record.detach().to(self.device)

        with torch.no_grad():
            all_output_1 = self.model(self.incor_data[:, 0, :], return_all_outputs=True)
            all_output_2 = self.model(self.incor_data[:, 1, :], return_all_outputs=True)
            all_output_nor = self.model(self.normal_data, return_all_outputs=True)
            pre_h1 = all_output_1[check_layer - 1]
            pre_h2 = all_output_2[check_layer - 1]
            pre_h_nor = all_output_nor[check_layer - 1]
            h_x1 = all_output_1[check_layer]
            h_x2 = all_output_2[check_layer]
            h_xnor = all_output_nor[check_layer]
            neuron_value_1 = h_x1[:, analyze_neuron]
            neuron_value_2 = h_x2[:, analyze_neuron]
            neuron_value_nor = h_xnor[:, analyze_neuron]

        in_weight_num = len(pre_h1[0])

        target_1 = torch.empty([self.incor_data_num, 2], requires_grad=False, device=self.device)
        target_2 = torch.empty([self.incor_data_num, 2], requires_grad=False, device=self.device)

        l_bounds_1 = endpoints_1[..., 0:2].min(dim=-1)[0]
        u_bounds_1 = endpoints_1[..., 2:4].max(dim=-1)[0]
        l_bounds_2 = endpoints_2[..., 0:2].min(dim=-1)[0]
        u_bounds_2 = endpoints_2[..., 2:4].min(dim=-1)[0]

        # shape of combined_cond - [N, k]
        result_expanded = result.unsqueeze(1).expand(-1, self.interval_num)
        cond_0 = (l_bounds_1 > 0) & (l_bounds_2 > 0)
        cond_1 = (u_bounds_1 < 0) & (u_bounds_2 < 0)
        combined_cond = torch.where(result_expanded == 0, cond_0, cond_1)
        # combined_cond = ((repair_1_min > 0) & (repair_2_min > 0)) | ((repair_1_max < 0) & (repair_2_max < 0))

        # shape of has_point_set - [N]
        has_point_set = combined_cond.any(dim=1)  # [incor_data_num]

        target_1 = neuron_value_1.unsqueeze(1).expand(-1, 2).clone()
        target_2 = neuron_value_2.unsqueeze(1).expand(-1, 2).clone()

        # consider the data donot have 'satisfied' interval
        if not has_point_set.all():
            max_left = torch.maximum(
                torch.minimum(endpoints_1[..., 0], endpoints_2[..., 0]),
                torch.minimum(-endpoints_1[..., 2], -endpoints_2[..., 2])
            )
            max_right = torch.maximum(
                torch.minimum(endpoints_1[..., 1], endpoints_2[..., 1]),
                torch.minimum(-endpoints_1[..., 3], endpoints_2[..., 3])
            )
            
            stacked_max = torch.stack([max_left, max_right], dim=2)  # [N, num_points, 2]
            max_per_point, indices = torch.max(stacked_max, dim=2)   # [N, num_points]
            max_per_sample, j_max = torch.max(max_per_point, dim=1)  # [N]
            
            batch_indices = torch.arange(self.incor_data_num, device=self.device)
            selected_eps = eps_record[batch_indices, j_max]  # shape of selected_eps - [N, 2]

            # NOTE test
            use_idx = (1 - indices[batch_indices, j_max]).long() # 0 - right or 1 - left
            
            row_indices = torch.arange(self.incor_data_num, device=self.device)
            selected_values = selected_eps[row_indices, use_idx]  # [N]

            # get "ideal interval [v*, v*]"
            new_vals = selected_values.unsqueeze(1).repeat(1, 2)  # [N, 2]

            update_mask = ~has_point_set
            # only update the data that donot have 'satisfied' interval
            target_1[update_mask] = new_vals[update_mask]
            target_2[update_mask] = new_vals[update_mask]

        # consider the data have 'satisfied' interval
        if has_point_set.any():
            point_mask = combined_cond
            # shape of d0 - [N, k]
            d0 = torch.abs(eps_record[..., 0] - neuron_value_1.unsqueeze(1))
            d1 = torch.abs(eps_record[..., 1] - neuron_value_1.unsqueeze(1))
            # shape of min_d0_d1 - [N, k]
            min_d0_d1 = torch.minimum(d0, d1)
            
            d2 = torch.abs(eps_record[..., 0] - neuron_value_2.unsqueeze(1))
            d3 = torch.abs(eps_record[..., 1] - neuron_value_2.unsqueeze(1))
            min_d2_d3 = torch.minimum(d2, d3)
            
            abs_diff = torch.abs(min_d0_d1 - min_d2_d3)
            
            # filter not satisfied intervals
            max_diff, opt_index = torch.max(abs_diff * point_mask, dim=1)
            
            batch_indices = torch.arange(self.incor_data_num, device=self.device)
            selected_points = eps_record[batch_indices, opt_index][:, :2]
            
            update_mask = has_point_set
            target_1[update_mask] = selected_points[update_mask]
            target_2[update_mask] = selected_points[update_mask]
        
        # ================== 修改开始 1：构造 Flag 标签 ==================
        flag_unfair_1 = torch.ones(len(pre_h1), 1, device=self.device)
        flag_unfair_2 = torch.ones(len(pre_h2), 1, device=self.device)
        flag_normal = torch.zeros(len(pre_h_nor), 1, device=self.device)
        mini_flags = torch.cat([flag_unfair_1, flag_unfair_2, flag_normal], dim=0)
        # ================== 修改结束 1 ==================
        
        target_1 = target_1.squeeze(1)
        target_2 = target_2.squeeze(1)

        mini_input_parts = [pre_h1.detach(), pre_h2.detach(), pre_h_nor.detach()]
        mini_label_parts = [target_1, target_2]
        mini_label_nor = neuron_value_nor.unsqueeze(1).expand(-1, 2)
        mini_label_parts.append(mini_label_nor)

        mini_input = torch.cat(mini_input_parts, dim=0)
        mini_label = torch.cat(mini_label_parts, dim=0)
        mini_data = TensorDataset(mini_input, mini_label, mini_flags)

        d = len(self.model.state_dict().keys()) - len(model.state_dict().keys())
        para_name = list(self.model.state_dict().keys())
        pre_check_wegiht = self.model.state_dict()[para_name[d - 2]]
        pre_check_bias = self.model.state_dict()[para_name[d - 1]]

        post_check_wegiht = self.model.state_dict()[para_name[d]]
        # post_check_bias = self.model.state_dict()[para_name[d + 1]]

        mini_loader = DataLoader(dataset=mini_data, batch_size=self.BATCH_SIZE, shuffle=True)

        check_neuron_in_weight = pre_check_wegiht[analyze_neuron]
        check_neuron_bias = pre_check_bias[analyze_neuron]

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
        # NOTE change
        for epoch in range(self.local_epoch):
            out_1 = mini_nn(pre_h1).squeeze(1)
            out_2 = mini_nn(pre_h2).squeeze(1)
            loss = ((out_1 - out_2) ** 2).mean()

            d1_0 = out_1 - target_1[:, 0]
            d1_1 = out_1 - target_1[:, 1]
            d2_0 = out_2 - target_1[:, 0]
            d2_1 = out_2 - target_1[:, 1]

            mask1 = (d1_0 > 0) & (d1_1 > 0)
            mask2 = (d1_0 < 0) & (d1_1 < 0)
            loss_11 = torch.where(mask1, d1_1 * d1_1, torch.zeros_like(d1_1))
            loss_12 = torch.where(mask2, d1_0 * d1_0, torch.zeros_like(d1_0))
            loss += (loss_11 + loss_12).mean()

            mask3 = (d2_0 > 0) & (d2_1 > 0)
            mask4 = (d2_0 < 0) & (d2_1 < 0)
            loss_21 = torch.where(mask3, d2_1 * d2_1, torch.zeros_like(d2_1))
            loss_22 = torch.where(mask4, d2_0 * d2_0, torch.zeros_like(d2_0))
            loss += (loss_21 + loss_22).mean()

            out_nor = mini_nn(pre_h_nor).squeeze(1)
            loss += ((neuron_value_nor - out_nor) ** 2).mean()

            if loss == 0:
                continue
            optim.zero_grad()
            loss.backward()
            optim.step()

        # for epoch in range(self.local_epoch):
        #     for step, (x, y, flag) in enumerate(mini_loader):
        #         output = mini_nn(x).squeeze(1)
        #         l0 = output - y[:, 0]
        #         l1 = output - y[:, 1]
                
        #         mask1 = (l0 > 0) & (l1 > 0)
        #         mask2 = (l0 < 0) & (l1 < 0)
        #         loss_elements = torch.where(mask1, l1 * l1, torch.zeros_like(l1)) + \
        #                         torch.where(mask2, l0 * l0, torch.zeros_like(l0))
                
        #         is_unfair = flag.view(-1).bool()
        #         is_normal = ~is_unfair
        #         loss = torch.tensor(0.0, device=self.device)
        #         if is_unfair.any():
        #             loss += loss_elements[is_unfair].mean()
        #         if is_normal.any():
        #             loss += loss_elements[is_normal].mean()
        #         if loss == 0:
        #             continue
        #         optim.zero_grad()
        #         loss.backward()
        #         optim.step()

        repaired_para = {}
        for key in self.model.state_dict().keys():
            repaired_para[key] = self.model.state_dict()[key]
        b = mini_nn.state_dict()['2.bias']
        w = mini_nn.state_dict()['2.weight'][0]
        repaired_para[para_name[d - 2]][analyze_neuron] = mini_nn.state_dict()['0.weight']
        repaired_para[para_name[d - 1]][analyze_neuron] = mini_nn.state_dict()['0.bias']
        repaired_para[para_name[d + 1]] += b * post_check_wegiht[:, analyze_neuron]
        repaired_para[para_name[d]][:, analyze_neuron] *= w

        self.repair_time += time.time() - t
        self.model.load_state_dict(repaired_para)