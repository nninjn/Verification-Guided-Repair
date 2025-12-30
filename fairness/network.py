import torch
import torch.nn as nn
import torch.nn.functional as F


def slice_model_at_penultimate_layer(original_model):
    # Slice the model by modifying its forward method to stop at the penultimate layer (i.e., remove the last linear layer).
    linear_layers = original_model.layers
    if len(linear_layers) < 2:
        raise ValueError("Model must have at least two linear layers.")
    layers = []
    for layer in linear_layers[: -1]:
        layers.append(layer)
        if isinstance(layer, nn.Linear):
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


model_configs = {
    'net1': [16, 8, 1],
    'net2': [30, 20, 15, 10, 5, 1]
}


class Model(nn.Module):
    def __init__(self, input_size, layers):
        super(Model, self).__init__()
        self.layers = nn.ModuleList()
        self.if_sig = True
        self.sigmoid = nn.Sigmoid()
        for i, layer_size in enumerate(layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, layer_size))
            else:
                self.layers.append(nn.Linear(layers[i-1], layer_size))
    
    def forward(self, x, return_all_outputs=False):
        all_outputs = [x, ]
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            all_outputs.append(x)
        x = self.layers[-1](x)
        all_outputs.append(x)
        if self.if_sig:
            x = self.sigmoid(x)
        if return_all_outputs:
            return all_outputs
        return x

    def get_all(self, x):
        output = [x]
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            output.append(x)
        x = self.layers[-1](x)
        output.append(x)
        return output
    
    def split(self, layer=1):
        split_layers = []
        for i in range(len(self.layers)):
            if i >= layer:
                split_layers.append(self.layers[i])
                if i < len(self.layers) - 1:
                    split_layers.append(nn.ReLU())
        return SplitModel(split_layers, self.if_sig), layer

class SplitModel(nn.Module):
    def __init__(self, layers, if_sig):
        super(SplitModel, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.if_sig = if_sig
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.if_sig:
            x = self.sigmoid(x)
        return x

def create_model(model_name, input_size):
    if model_name in model_configs:
        return Model(input_size, model_configs[model_name])
    else:
        raise ValueError(f"Model {model_name} not found in configurations.")
