from .common import PositionalEncoding
from mymodels import TransformerEncoderLayer
from mymodels import TransformerEncoder
from mymodels import ModuleDict
import jittor as jt
import jittor.nn as nn
class StyleEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model parameters
        self.motion_coef_dim = 50
        if args.rot_repr == 'aa':
            self.motion_coef_dim += 1 if args.no_head_pose else 4
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

        self.feature_dim = args.feature_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.mlp_ratio = args.mlp_ratio

        # Transformer for feature extraction
        encoder_layer = TransformerEncoderLayer(
            d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu', dropout=0.1
        )

        self.PE = PositionalEncoding(self.feature_dim)

        encoder_layer_class = lambda: TransformerEncoderLayer(
        d_model=self.feature_dim, nhead=self.n_heads, 
        dim_feedforward=self.mlp_ratio * self.feature_dim, 
        activation='gelu', dropout=0.1
        )

        

        self.encoder = ModuleDict({
            'motion_proj': nn.Linear(self.motion_coef_dim, self.feature_dim),
            'transformer': TransformerEncoder(encoder_layer_class, num_layers=self.n_layers),
        })


    @property
    def device(self):
        return 'cuda' if jt.flags.use_cuda else 'cpu'

    def execute(self, motion_coef):
        """
        :param motion_coef: (batch_size, seq_len, motion_coef_dim)
        :param audio: (batch_size, seq_len)
        :return: (batch_size, feature_dim)
        """
        batch_size, seq_len, _ = motion_coef.shape

        # Motion
        motion_feat = self.encoder['motion_proj'](motion_coef)
        motion_feat = self.PE(motion_feat)

        feat = self.encoder['transformer'](motion_feat)  # (N, L, feat_dim)
        feat = feat.mean(dim=1)  # Pooling to (N, feat_dim)

        return feat
