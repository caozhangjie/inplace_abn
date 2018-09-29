import torch
import torch.nn as nn

class GraphCNN(nn.Module):
    def __init__(self, adjacency_m, feature_dim, dim_list):
        self.object_class = [19,20,51,52,53,54,55,56,57,58,59,60,61,61,63]
        self.activation = nn.ReLU()
        self.layers = []
        cur_dim = feature_dim
        for dim in dim_list:
            self.layers.append(nn.Linear(cur_dim, dim, bias=False))
            cur_dim = dim
        self.weight_layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        adjacency_m = torch.zeros([x.size(0), x.size(0)])
        for i in self.object_class + range(64, x.size(0)):
            for j in self.object_class + range(64, x.size(0)):
                adjacency_m[i, j] = 1.0
                adjacency_m[j, i] = 1.0
            for j in range(64):
                if j not in self.object_class:
                    adjacency_m[i, j] = 1.0
                    adjacency_m[j, i] = 1.0
        self.d_matrix = torch.diag(1.0 / torch.rsqrt(torch.abs(torch.sum(adjacency_m, dim=1))))
        self.norm_adjacency_m = torch.matmul(self.d_matrix, torch.matmul(self.adjacency_m, self.d_matrix))
        #bs = x.size(0)
        for layer in self.layers:
            x = torch.matmul(self.norm_adjacency_m, torch.matmul(layer(x)))
            x = self.activation(x)
        return x
