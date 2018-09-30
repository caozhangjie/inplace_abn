import torch
import torch.nn as nn

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    

class GraphCNN(nn.Module):
    def __init__(self, feature_dim, dim_list):
        self.object_class = [19,20,51,52,53,54,55,56,57,58,59,60,61,61,63]
        self.activation = nn.ReLU()
        self.layers = []
        self.norm_layers = []
        cur_dim = feature_dim
        for dim in dim_list:
            self.layers.append(nn.Linear(cur_dim, dim, bias=False))
            cur_dim = dim
            self.norm_layers = nn.BatchNorm1d(dim)
        self.weight_layers = nn.Sequential(*(self.layers+self.norm_layers))
        self.weight_layers.apply(init_weights)
        
    def forward(self, x):
        seq_len = x.size(1)
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
        for i in range(len(self.layers)):
            x = torch.matmul(self.norm_adjacency_m, self.layers[i](x.view(-1, x.size(2)).contiguous()).view(x.size(0), -1))
            x = self.norm_layers[i](x)
            x = self.activation(x)
            x = x.view(x.size(0), seq_len, -1).contiguous()
        return x
