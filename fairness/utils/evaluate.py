import torch
import random
import itertools
import numpy as np
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference


def my_dp(model, X_test, y_test, A_test):
    pred = np.int64(model(X_test).cpu().detach().numpy() > 0.5)
    new_dp = demographic_parity_difference(y_test, pred, sensitive_features=A_test)
    return new_dp

def my_eo(model, X_test, y_test, A_test):
    output = model(X_test)
    pred = np.int64(output.cpu().detach().numpy() > 0.5)
    eo = equalized_odds_difference(y_test, pred, sensitive_features=A_test)
    return eo
    
from .data_config import *
def idi_test(dataset, model, sen, base_data, unfair_num=None, get_data=False, use_base_data=False, gen_num=100000):
    if hasattr(model, 'if_sig'):
        model.if_sig = True
    model.eval()
    device = next(model.parameters()).device
    dataset_config = {
        "adult": adult,
        "bank": bank,
        "census": census,
        "compas": compas,
        "credit": credit,
        "default": default,
        "diabetes": diabetes,
        "heart": heart,
        "students": students,
        "meps15": meps15,
        "meps16": meps16
    }[dataset]
    bounds = dataset_config.input_bounds
    
    sensitive_indices = [sen] if isinstance(sen, int) else list(sen)
    protect_domain = [list(range(int(bounds[idx][0]), int(bounds[idx][1]) + 1)) for idx in sensitive_indices]
    # print('----')
    # print(protect_domain)
    # print('----')
    all_combs = np.array(list(itertools.product(*protect_domain)), dtype=np.float32)
    sensitive_comb_tensor = torch.tensor(all_combs, dtype=torch.float32).to(device)
    num_cases = len(all_combs)
    data_tensor = base_data.clone().detach().to(device)
    if use_base_data == False:
        data_tensor = torch.tensor([
            [random.randint(*bound) for bound in bounds]
            for _ in range(gen_num)#
        ], dtype=torch.float32).to(device)
    total_samples, num_features = data_tensor.shape
    
    variants = data_tensor.unsqueeze(1)                     # [N, 1, F]
    variants = variants.expand(-1, num_cases, -1).clone()   # [N, C, F]
    
    #[1, C, S] -> [N, C, S]
    sensitive_comb_expanded = sensitive_comb_tensor.unsqueeze(0).expand(total_samples, -1, -1)
    variants[:, :, sensitive_indices] = sensitive_comb_expanded
    batch_variants = variants.reshape(-1, num_features)#[N*C, F]
    with torch.no_grad():
        outputs = model(batch_variants)
        preds = (outputs > 0.5).int().view(total_samples, num_cases)  # [N, C]
    all_same = (preds == preds[:, [0]]).all(dim=1)
    not_fair = (~all_same).sum().item()
    
    unfair_rate = not_fair / total_samples if total_samples > 0 else 0.0
    if(not get_data):
        return unfair_rate

    unfair_indices = torch.where(~all_same)[0]

    unfair_pairs = []
    for i in unfair_indices:
        sample_variants = variants[i]#[num_cases, F]
        sample_preds = preds[i]  # [num_cases]
        correct_mask = (sample_preds == 1).squeeze()
        wrong_mask = ~correct_mask
        
        correct_set = sample_variants[correct_mask]  # [C_correct, F]
        wrong_set = sample_variants[wrong_mask]      # [C_wrong, F]
        
        if correct_set.shape[0] == 0 or wrong_set.shape[0] == 0:
            continue

        for correct_i in correct_set:
            for wrong_i in wrong_set:
                pair = torch.stack([correct_i, wrong_i])  # [2, F]
                unfair_pairs.append(pair.cpu().numpy())
    unfair_X = np.array(unfair_pairs) if len(unfair_pairs) > 0 else np.array([])

    if unfair_num is not None and unfair_X.shape[0] > unfair_num:
        indices = np.random.choice(len(unfair_X), unfair_num, replace=False)
        unfair_X = unfair_X[indices]
    
    total_counter = unfair_X.shape[0]

    if len(unfair_pairs) > 0:
        unfair_X_np = np.array(unfair_pairs)
        if unfair_num is not None and unfair_X_np.shape[0] > unfair_num:
            indices = np.random.choice(len(unfair_X_np), unfair_num, replace=False)
            unfair_X_np = unfair_X_np[indices]
        total_counter = unfair_X_np.shape[0]
        flattened_data = unfair_X_np.reshape(-1, num_features)
        unfair_data_tensor = torch.from_numpy(flattened_data).float().to(device)
    
    return unfair_rate, unfair_X, total_counter, unfair_data_tensor

def my_idi_test(sample_round, num_gen, dataset, model, sen, device):
    model.eval()
    dataset_config = {
        "adult": adult,
        "bank": bank,
        "census": census,
        "compas": compas,
        "credit": credit,
        "default": default,
        "diabetes": diabetes,
        "heart": heart,
        "students": students,
        "meps15": meps15,
        "meps16": meps16
    }[dataset]

    bounds = dataset_config.input_bounds

    sensitive_indices = [sen] if isinstance(sen, int) else list(sen)
    protect_domain = []
    for sensitive_idx in sensitive_indices:
        lower, upper = bounds[sensitive_idx]
        protect_domain.append(list(range(int(lower), int(upper) + 1)))
    
    all_combs = np.array(list(itertools.product(*protect_domain)), dtype=np.float32)
    num_cases = len(all_combs)

    not_fair = 0
    collected_unfair_tensors = []

    for ii in range(sample_round):
        X = torch.tensor([
            [random.randint(int(lower), int(upper)) for lower, upper in bounds]
            for _ in range(num_gen)
        ], dtype=torch.float32)

        # X_similar shape: (num_gen, num_cases, num_features)
        X_similar = X.unsqueeze(1).repeat(1, num_cases, 1)
        X_similar[:, :, sensitive_indices] = torch.tensor(all_combs, dtype=torch.float32)
        
        batch_size, num_cases, num_features = X_similar.shape
        
        with torch.no_grad():
            X_flat = X_similar.view(-1, num_features).to(device)
            y = model(X_flat)
            pred = (y > 0.5).int().view(batch_size, num_cases)
        
        # pred [batch_size, num_cases]
        all_same = (pred == pred[:, 0].unsqueeze(1)).all(dim=1)
        current_unfair_mask = ~all_same
        not_fair += current_unfair_mask.sum().item()

        if current_unfair_mask.any():
            unfair_indices = torch.where(current_unfair_mask)[0]
            
            # shape: [num_unfair, num_cases, num_features]
            bad_variants = X_similar[unfair_indices.cpu()]
            # shape: [num_unfair, num_cases]
            bad_preds = pred[unfair_indices]

            for i in range(len(unfair_indices)):
                sample_variants = bad_variants[i] # [num_cases, F]
                sample_p = bad_preds[i]           # [num_cases]

                group_0 = sample_variants[(sample_p == 0).cpu()]
                group_1 = sample_variants[(sample_p == 1).cpu()]

                if len(group_0) > 0 and len(group_1) > 0:
                    p0 = group_0.repeat_interleave(len(group_1), dim=0) 
                    p1 = group_1.repeat(len(group_0), 1)

                    pairs = torch.stack((p0, p1), dim=1)
                    
                    flat_pairs = pairs.view(-1, num_features)
                    
                    collected_unfair_tensors.append(flat_pairs)
        # ------ 结束 ------

    total_samples = sample_round * num_gen
    unfair_rate = not_fair / total_samples

    if len(collected_unfair_tensors) > 0:
        # 拼接成一个大的 Tensor: [总对数*2, features]
        unfair_data_tensor = torch.cat(collected_unfair_tensors, dim=0)
    else:
        unfair_data_tensor = torch.empty((0, len(bounds)), device=device)

    return unfair_rate, unfair_data_tensor

# def my_idi_test(sample_round, num_gen, dataset, model, sen, device):
#     model.eval()
#     dataset_config = {
#         "adult": adult,
#         "bank": bank,
#         "census": census,
#         "compas": compas,
#         "credit": credit,
#         "default": default,
#         "diabetes": diabetes,
#         "heart": heart,
#         "students": students,
#         "meps15": meps15,
#         "meps16": meps16
#     }[dataset]
#     # if dataset == "adult" or dataset == "bank":
#     #     constraint_path = f'/data/home/mjnn/majianan/fairness-repair/chenwei/fairness/data/PGD_dataset/{dataset}/constraint.npy'
#     #     bounds = np.load(constraint_path)
#     # else:
#     #     bounds = dataset_config.input_bounds
#     bounds = dataset_config.input_bounds

#     sensitive_indices = [sen] if isinstance(sen, int) else list(sen)
#     protect_domain = []
#     for sensitive_idx in sensitive_indices:
#         lower, upper = bounds[sensitive_idx]
#         protect_domain.append(list(range(lower, upper + 1)))
    
#     all_combs = np.array(list(itertools.product(*protect_domain)), dtype=np.float32)
#     num_cases = len(all_combs)

#     not_fair = 0
#     unfair_samples = []

#     for ii in range(sample_round):
#         X = torch.tensor([
#             [random.randint(int(lower), int(upper)) for lower, upper in bounds]
#             for _ in range(num_gen)
#         ], dtype=torch.float32)

#         # shape of X_similar (num_gen, num_cases, num_features)
#         X_similar = X.unsqueeze(1).repeat(1, num_cases, 1)
#         X_similar[:, :, sensitive_indices] = torch.tensor(all_combs, dtype=torch.float32)
#         batch_size, num_cases, _ = X_similar.shape
#         with torch.no_grad():
#             X_flat = X_similar.view(-1, len(bounds)).to(device)
#             y = model(X_flat)
#             pred = (y > 0.5).int().view(batch_size, num_cases)
#         all_same = (pred == pred[:, 0].unsqueeze(1)).all(dim=1)
#         current_unfair = ~all_same
#         not_fair += current_unfair.sum().item()

#     total_samples = sample_round * num_gen
#     unfair_rate = not_fair / total_samples
#     unfair_X = np.array(unfair_samples, dtype=float) if unfair_samples else np.array([])

#     return unfair_rate, unfair_X


def test_acc(model, X_test, y_test):
    # if hasattr(model, 'if_sig'):
    #     model.if_sig = True
    model.eval()
    y = model(X_test)
    pred = torch.where(y > 0.5, 1, 0).squeeze().cpu().numpy()
    assert pred.shape == y_test.shape, f"error in shape, {pred.shape=} {y_test.shape=}"
    correct = (pred == y_test).sum()
    acc = correct / len(y_test)
    return acc

def evaluation(model, dataset, X_test, y_test, sen, device, verbose=True, unfair_data_tensor=None, unfair_data_tensor2=None):
    if hasattr(model, 'if_sig'):
        model.if_sig = True
    model.eval()
    X_test = torch.from_numpy(X_test).to(device).float()
    acc = test_acc(model, X_test, y_test)
    return_data = 0
    if(unfair_data_tensor==None):
        idi_per_data, _, _, unfair_data_tensor= idi_test(dataset, model, sen, X_test, get_data=True)
        idi_per, unfair_data_tensor2 = my_idi_test(10, 10000, dataset, model, sen, device)
        return_data = 1
    else:
        idi_per_data = idi_test(dataset, model, sen, unfair_data_tensor, use_base_data=True)
        idi_per = idi_test(dataset, model, sen, unfair_data_tensor2, use_base_data=True)
        print("wwwwww")

    if verbose:
        print("Evaluation results:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Individual Fairness Violation Rate (IDI)'{sen}': {idi_per:.4f}")
        print(f"Individual Fairness Violation Rate (IDI) of Dataset'{sen}': {idi_per_data:.4f}")
    if(return_data):
        return acc, 1, 1, unfair_data_tensor, unfair_data_tensor2
    else:
        return acc, idi_per, idi_per_data