import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionRegression(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EmotionRegression, self).__init__()
        input_dim = args[0]
        hidden_dim = args[1]
        num_layers = args[2]
        output_dim = args[3]
        p = kwargs.get("dropout", 0.5)

        self.fc=nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p)
            )
        ])
        for lidx in range(num_layers-1):
            self.fc.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(p)
                )
            )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.inp_drop = nn.Dropout(p)
    def get_repr(self, x):
        h = self.inp_drop(x)
        for lidx, fc in enumerate(self.fc):
            h=fc(h)
        return h
    
    def forward(self, x):
        h=self.get_repr(x)
        result = self.out(h)
        return result