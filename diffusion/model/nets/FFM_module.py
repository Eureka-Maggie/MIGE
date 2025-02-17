import torch
import torch.nn as nn
import re
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math

class FFM(nn.Module):
    def __init__(
            self,
            embed_dim=768,
            num_heads= 8,
            kv_dim=768,
            scale_factor=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.q_proj_1 = nn.Linear(kv_dim, embed_dim, bias=False)

        k_modules = [nn.Linear(1152, 768)]
        for _ in range(1,2):
            k_modules.append(nn.GELU())
            k_modules.append(nn.Linear(768, 768))
        self.k_proj_1 = nn.Sequential(*k_modules)

        v_modules = [nn.Linear(1152, 768)]
        for _ in range(1,2):
            v_modules.append(nn.GELU())
            v_modules.append(nn.Linear(768, 768))
        self.v_proj_1 = nn.Sequential(*v_modules)

        self.ln_q_1 = norm_layer(self.embed_dim)
        self.ln_k_1 = norm_layer(self.embed_dim)
        self.ln_v_1 = norm_layer(self.embed_dim)

        self.ln_out = norm_layer(self.embed_dim)

        self.clip_attn = nn.MultiheadAttention(embed_dim, num_heads)

        self.mlp = nn.Linear(self.embed_dim,self.embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.mlp.weight, 0)
        nn.init.constant_(self.mlp.bias, 0)

    def forward(self, clip, vae, attn_mask=None):

        x_multi = vae ## key,value
        x = clip #[3,32,768] query

        residual_0 = x
        
        key = self.ln_k_1(self.k_proj_1(x_multi)).permute(1, 0, 2) #[257,3,768]
        value = self.ln_v_1(self.v_proj_1(x_multi)).permute(1, 0, 2)
        query = self.ln_q_1(self.q_proj_1(x)).permute(1, 0, 2) #[32,3,768]

        attn_out = self.clip_attn(
            query,
            key,
            value,
            attn_mask=attn_mask)[0]
        x = self.mlp(attn_out).permute(1, 0, 2)
        x = x + residual_0

        return self.ln_out(x)
