import torch
import torch.nn as nn 
class PoseEstimator(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.dim_reduce = nn.Sequential(nn.Linear(1024,512), nn.SiLU(), nn.LayerNorm(512), nn.Linear(512, 256), nn.SiLU(), nn.LayerNorm(256), nn.Linear(256,128), nn.SiLU(), nn.LayerNorm(128), nn.Linear(128,64))
        self.final_mlp = nn.Sequential(nn.Linear(25088, 128), nn.SiLU(), nn.LayerNorm(128),nn.Linear(128,7))
        self.num_layers = num_layers
        self.layers = nn.ModuleList([AttentionLayer() for i in range(num_layers)])
    def forward(self,features): # (2, 201, 1024)
        fa, fb= features[:,5:,:].chunk(2, dim = 0)
        for i in range(self.num_layers):
            fa, fb = self.layers[i](fa, fb)
        total = torch.cat([fa.squeeze(0),fb.squeeze(0)], dim = 0)
        total = self.dim_reduce(total)

        return self.final_mlp(total.flatten())


class AttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(1024,16, batch_first = True)
        self.cross_attention = nn.MultiheadAttention(1024,16, batch_first = True)
        self.norm1  = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(1024)
        self.norm3 = nn.LayerNorm(1024)
        self.norm4 = nn.LayerNorm(1024)
        self.ffn1 = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
        self.ffn2 = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
    def forward(self, fa, fb): # 1, 196, 1024
        stacked = torch.cat([fa,fb], dim = 0)
        norm = self.norm1(stacked)
        stacked = stacked + self.self_attention(norm, norm, norm)[0]
        stacked = stacked + self.ffn1(self.norm2(stacked))

        fa, fb = self.norm3(stacked).chunk(2, dim = 0)

        q = torch.cat([fa,fb], dim = 0)
        kv = torch.cat([fb, fa], dim = 0)

        x = self.cross_attention(q, kv, kv)[0]
        stacked = stacked + x
        stacked = stacked + self.ffn2(self.norm4(stacked))

        return stacked.chunk(2, dim = 0)
