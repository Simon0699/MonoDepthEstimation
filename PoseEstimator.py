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
        self.self_attention = nn.MultiheadAttention(1024,16)
        self.cross_attention = nn.MultiheadAttention(1024,16)
        self.norm1  = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(1024)
        self.norm3 = nn.LayerNorm(1024)
        self.norm4 = nn.LayerNorm(1024)
        self.ffn = nn.Sequential(nn.Linear(1024, 2048), nn.SiLU(), nn.LayerNorm(2048), nn.Linear(2048, 1536), nn.SiLU(), nn.LayerNorm(1536), nn.Linear(1536, 1024))
    def forward(self, fa, fb): # 1, 196, 1024
        stacked = torch.cat([fa,fb], dim = 0)
        print(stacked.shape)
        stacked = self.norm1(stacked)
        stacked = stacked + self.self_attention(stacked, stacked, stacked)[0]
        stacked = self.norm2(stacked)
        stacked = stacked + self.ffn(stacked)
        stacked = self.norm3(stacked)

        fa, fb = stacked.chunk(2, dim = 0)

        q = torch.cat([fa,fb], dim = 0)
        kv = torch.cat([fb, fa], dim = 0)

        x = self.cross_attention(q, kv, kv)[0]
        stacked = stacked + x
        stacked = self.norm4(stacked)
        stacked = stacked + self.ffn(stacked)

        return stacked.chunk(2, dim = 0)
