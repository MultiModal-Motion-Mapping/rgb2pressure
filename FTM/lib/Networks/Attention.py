import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe
 
    def forward(self, x):
        device = x.device
        self.pe = self.pe.to(device=device)
        
        x = x + self.pe[:, :x.size(1)]
        return x
    
class TokenEmbedding(nn.Module):
    def __init__(self, indim: int = 512, seqlen_max: int = 500):
        super().__init__()
        self.positions = nn.Parameter(torch.randn(seqlen_max , indim))
        
    def forward(self, x):
        b, seqlen, indim = x.shape
        
        return x+self.positions[:seqlen, :indim].unsqueeze(0)

        
class Attention(nn.Module):
    def __init__(self, indim:int=3072, emb_size:int=3072, expansion:int=2, drop_p:int=0.1):
        super().__init__()
        
        self.tokenemb = TokenEmbedding(indim=indim)
        
        self.key = nn.Linear(indim, emb_size)
        self.query = nn.Linear(indim, emb_size)
        self.value = nn.Linear(indim, emb_size)
        self.multihead = nn.MultiheadAttention(embed_dim=emb_size, num_heads=8, dropout=drop_p, bias=False)
        
        self.FFN = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),   
        )
        
        self.layernorm = nn.LayerNorm(3072)  
        

        
    def forward(self, x):
        b, seqlen, indim = x.shape
        
        token = self.tokenemb(x)
        
        # Attention
        attn_in = self.layernorm(token)
        key = self.key(attn_in)
        query = self.query(attn_in)
        value = self.value(attn_in)
        attn_out, _ = self.multihead(key=key, query=query, value=value)
        
        res1_out = token+attn_out
        
        FFN_in = self.layernorm(res1_out)
        FFN_out = self.FFN(FFN_in)
        
        res2_out = FFN_out+res1_out
    
        return res2_out
        