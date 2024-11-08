import jittor as jt
import jittor.nn as nn
from jittor import attention
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0, activation='gelu'):
        super(TransformerEncoderLayer, self).__init__()
        

        self.self_attn = jt.attention.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        if activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation function {activation}")

    def execute(self, src):
        src2, _ = self.self_attn(src, src, src)
        src = src + src2
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + src2
        src = self.norm2(src)

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer_class, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([encoder_layer_class() for _ in range(num_layers)])
    
    def execute(self, src):
        for layer in self.layers:
            src = layer(src)  
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, activation='gelu', batch_first=False):
        super(TransformerDecoderLayer, self).__init__()

        self.batch_first = batch_first

        self.self_attn = jt.attention.MultiheadAttention(d_model, nhead)
        self.multihead_attn = jt.attention.MultiheadAttention(d_model, nhead)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        if activation == 'gelu':
            self.activation = nn.gelu
        elif activation == 'relu':
            self.activation = nn.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def execute(self, tgt, memory, tgt_mask=None, memory_mask=None):
     
        if self.batch_first:
            tgt = jt.transpose(tgt, 0, 1)
            memory = jt.transpose(memory, 0, 1)

        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.activation(self.linear1(tgt)))
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)

        if self.batch_first:
            tgt = jt.transpose(tgt, 0, 1)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer_class, num_layers):
        super(TransformerDecoder, self).__init__()
        #self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.layers = nn.ModuleList([decoder_layer_class() for _ in range(num_layers)])
   
        
    def execute(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        return output




