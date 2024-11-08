import math
import jittor.nn as nn
import jittor as jt
import jittor as jt
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # vanilla sinusoidal encoding
        pe = jt.zeros(max_len, d_model)
        position = jt.arange(0, max_len, dtype=jt.float32).unsqueeze(1)
        div_term = jt.exp(jt.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = jt.sin(position * div_term)
        pe[:, 1::2] = jt.cos(position * div_term)
        pe = pe.unsqueeze(0)
        #self.register_buffer('pe', pe)
        self.pe = pe
    def execute(self, x):
        x = x + self.pe[:, x.shape[1], :]
        return self.dropout(x)


def enc_dec_mask(T, S, frame_width=2, expansion=0, device='cuda'):
    mask = jt.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device)


def pad_audio(audio, audio_unit=320, pad_threshold=80):
    batch_size, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = jt.nn.pad(audio, (reflect_len, reflect_len), mode='reflect')
            audio = jt.nn.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = jt.nn.pad(audio, (1, 1), mode='replicate')

    return audio
