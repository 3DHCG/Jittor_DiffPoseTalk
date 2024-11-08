import jittor.nn as nn
import jittor as jt
from .common import PositionalEncoding, enc_dec_mask, pad_audio

class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=0.02, s=0.008):
        super().__init__()

        if mode == 'linear':
            betas = jt.linspace(beta_1, beta_T, num_steps)
        elif mode == 'quadratic':
            betas = jt.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == 'sigmoid':
            betas = jt.sigmoid(jt.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1
        elif mode == 'cosine':
            steps = num_steps + 1
            x = jt.linspace(0, num_steps, steps)
            import math
            alpha_bars = jt.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            def jittor_clip(input_tensor, min_val=None, max_val=None):
                if min_val is not None:
                    input_tensor = jt.maximum(input_tensor, min_val)
                if max_val is not None:
                    input_tensor = jt.minimum(input_tensor, max_val)
                return input_tensor
            betas = jittor_clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f'Unknown diffusion schedule {mode}!')
        betas = jt.concat([jt.zeros(1), betas], dim=0)  # Padding beta_0 = 0

        alphas = 1 - betas
        log_alphas = jt.log(alphas)
        for i in range(1, log_alphas.shape[0]):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = jt.sqrt(betas)
        sigmas_inflex = jt.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.shape[0]):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = jt.sqrt(sigmas_inflex)

        self.num_steps = num_steps
        self.betas = jt.array(betas)
        self.alphas = jt.array(alphas)
        self.alpha_bars = jt.array(alpha_bars)
        self.sigmas_flex = jt.array(sigmas_flex)
        self.sigmas_inflex = jt.array(sigmas_inflex)
    def uniform_sample_t(self, batch_size):
        ts = jt.randint(1, self.num_steps + 1, (batch_size,))
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DiffTalkingHead(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Model parameters
        self.target = args.target
        self.architecture = args.architecture
        self.use_style = args.style_enc_ckpt is not None

        self.motion_feat_dim = 50
        if args.rot_repr == 'aa':
            self.motion_feat_dim += 1 if args.no_head_pose else 4
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

        self.fps = args.fps
        self.n_motions = args.n_motions
        self.n_prev_motions = args.n_prev_motions
        if self.use_style:
            self.style_feat_dim = args.d_style

        # Audio encoder
        
        self.audio_model = args.audio_model
        if self.audio_model == 'wav2vec2':
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')
            # wav2vec 2.0 weights initialization
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert':
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained('facebook/hubert-base-ls960')
            self.audio_encoder.feature_extractor._freeze_parameters()

            frozen_layers = [0, 1]
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')
        
        if args.architecture == 'decoder':
            self.audio_feature_map = nn.Linear(768, args.feature_dim)
            self.start_audio_feat = nn.Parameter(jt.randn(1, self.n_prev_motions, args.feature_dim))
        else:
            raise ValueError(f'Unknown architecture {args.architecture}!')

        self.start_motion_feat = nn.Parameter(jt.randn(1, self.n_prev_motions, self.motion_feat_dim))

        # Diffusion model
        #self.denoising_net = DenoisingNetwork(args, device)
        self.denoising_net = DenoisingNetwork(args)
        # diffusion schedule
        self.diffusion_sched = DiffusionSchedule(args.n_diff_steps, args.diff_schedule)

        # Classifier-free settings
        self.cfg_mode = args.cfg_mode
        guiding_conditions = args.guiding_conditions.split(',') if args.guiding_conditions else []
        self.guiding_conditions = [cond for cond in guiding_conditions if cond in ['style', 'audio']]
        if 'style' in self.guiding_conditions:
            if not self.use_style:
                raise ValueError('Cannot use style guiding without enabling it!')
            self.null_style_feat = nn.Parameter(jt.randn(1, 1, self.style_feat_dim))
        if 'audio' in self.guiding_conditions:
            audio_feat_dim = args.feature_dim
            self.null_audio_feat = nn.Parameter(jt.randn(1, 1, audio_feat_dim))
        for param in self.parameters():
            param.requires_grad = True
        for name, param in self.audio_encoder.named_parameters():
                if name.startswith("feature_projection"):
                    param.requires_grad = False
                if name.startswith("feature_extractor"):
                    param.requires_grad = False
                if name.startswith("encoder.layers"):
                    layer = int(name.split(".")[2])
                    if layer in frozen_layers:
                        param.requires_grad = False
        self.diffusion_sched.betas.stop_grad()
        self.diffusion_sched.alphas.stop_grad()
        self.diffusion_sched.alpha_bars.stop_grad()
        self.diffusion_sched.sigmas_flex.stop_grad()
        self.diffusion_sched.sigmas_inflex.stop_grad()
        self.denoising_net.TE.pe.stop_grad()
        
    @property
    def device(self):
        return 'cuda' if jt.flags.use_cuda else 'cpu'

    def execute(self, motion_feat, audio_or_feat, shape_feat, style_feat=None,
                prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_coef) motion coefficients or features
            audio_or_feat: (N, L_audio) raw audio or audio feature
            shape_feat: (N, d_shape) or (N, 1, d_shape)
            style_feat: (N, d_style)
            prev_motion_feat: (N, n_prev_motions, d_motion) previous motion coefficients or feature
            prev_audio_feat: (N, n_prev_motions, d_audio) previous audio features
            time_step: (N,)
            indicator: (N, L) 0/1 indicator of real (unpadded) motion coefficients

        Returns:
           motion_feat_noise: (N, L, d_motion)
        """
        if self.use_style:
            assert style_feat is not None, 'Missing style features!'

        batch_size = motion_feat.shape[0]

        if audio_or_feat.ndim == 2:
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
        audio_feat = audio_feat_saved.clone()

        if shape_feat.ndim == 2:
            shape_feat = shape_feat.unsqueeze(1)  # (N, 1, d_shape)
        if style_feat is not None and style_feat.ndim == 2:
            style_feat = style_feat.unsqueeze(1)  # (N, 1, d_style)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        # Classifier-free guidance
        if len(self.guiding_conditions) > 0:
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                if 'style' in self.guiding_conditions:
                    mask_style = jt.rand(batch_size) < null_cond_prob
                    style_feat = jt.where(mask_style.view(-1, 1, 1),
                                             self.null_style_feat.expand(batch_size, -1, -1),
                                             style_feat)
                if 'audio' in self.guiding_conditions:
                    mask_audio = jt.rand(batch_size) < null_cond_prob
                    audio_feat = jt.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)
            else:
                # len(self.guiding_conditions) > 1 and self.cfg_mode == 'incremental'
                # full (0.45), w/o style (0.45), w/o style or audio (0.1)
                mask_flag = jt.rand(batch_size)
                if 'style' in self.guiding_conditions:
                    mask_style = mask_flag > 0.55
                    style_feat = jt.where(mask_style.view(-1, 1, 1),
                                             self.null_style_feat.expand(batch_size, -1, -1),
                                             style_feat)
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag > 0.9
                    audio_feat = jt.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)

        if style_feat is None:
            # The model only accepts audio and shape features, i.e., self.use_style = False
            person_feat = shape_feat
        else:
            person_feat = jt.concat([shape_feat, style_feat], dim=-1)

        if time_step is None:
            # Sample time step
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)  # (N,)

        # The forward diffusion process
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]  # (N,)
        c0 = jt.sqrt(alpha_bar).view(-1, 1, 1)  # (N, 1, 1)
        c1 = jt.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (N, 1, 1)

        eps = jt.randn_like(motion_feat)  # (N, L, d_motion)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        # The reverse diffusion process
        motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat, person_feat,
                                                prev_motion_feat, prev_audio_feat, time_step, indicator)

        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions

        # # Strategy 1: resample during audio feature extraction
        # hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num).last_hidden_state  # (N, L, 768)

        # Strategy 2: resample after audio feature extraction (BackResample)
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
                                           frame_num=frame_num * 2).last_hidden_state  # (N, 2L, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        def linear_interpolate_1d(features, output_len):
            input_len = features.shape[-1]
            if input_len == output_len:
                return features
            scale_factor = (input_len - 1) / (output_len - 1)
            output_indices = jt.float32([i * scale_factor for i in range(output_len)])
            left_indices = jt.floor(output_indices).int32()
            right_indices = jt.ceil(output_indices).int32()
            right_indices = jt.minimum(right_indices, input_len - 1)

            left_weight = 1.0 - (output_indices - left_indices)
            right_weight = 1.0 - left_weight

            output_features = features[:, :, left_indices] * left_weight.unsqueeze(0).unsqueeze(0) + \
                              features[:, :, right_indices] * right_weight.unsqueeze(0).unsqueeze(0)

            return output_features
        hidden_states = linear_interpolate_1d(hidden_states, frame_num)  # (N, 768, L)
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)

        audio_feat = self.audio_feature_map(hidden_states)  # (N, L, feature_dim)
        return audio_feat

    @jt.no_grad()
    def sample(self, audio_or_feat, shape_feat, style_feat=None, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False):
        # Check and convert inputs
        batch_size = audio_or_feat.shape[0]

        # Check CFG conditions
        if cfg_mode is None:  # Use default CFG mode
            cfg_mode = self.cfg_mode
        if cfg_cond is None:  # Use default CFG conditions
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'style']]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)

        # sort cfg_cond and cfg_scale
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'style'].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []

        if 'style' in cfg_cond:
            assert self.use_style and style_feat is not None

        if self.use_style:
            if style_feat is None:  # use null style feature
                style_feat = self.null_style_feat.expand(batch_size, -1, -1)
        else:
            assert style_feat is None, 'This model does not support style feature input!'

        if audio_or_feat.ndim == 2:
            # Extract audio features
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)  # (N, L, feature_dim)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if shape_feat.ndim == 2:
            shape_feat = shape_feat.unsqueeze(1)  # (N, 1, d_shape)
        if style_feat is not None and style_feat.ndim == 2:
            style_feat = style_feat.unsqueeze(1)  # (N, 1, d_style)

        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)  # (N, n_prev_motions, d_motion)
        if prev_audio_feat is None:
            # (N, n_prev_motions, feature_dim)
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)

        if motion_at_T is None:
            import jittor as jt
            motion_at_T = jt.randn((batch_size, self.n_motions, self.motion_feat_dim))

        # Prepare input for the reverse diffusion process (including optional classifier-free guidance)
        if 'audio' in cfg_cond:
            audio_feat_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        else:
            audio_feat_null = audio_feat

        if 'style' in cfg_cond:
            import jittor as jt
            person_feat_null = jt.concat([shape_feat, self.null_style_feat.expand(batch_size, -1, -1)], dim=-1)
        else:
            if self.use_style:
                import jittor as jt
                person_feat_null = jt.concat([shape_feat, style_feat], dim=-1)
            else:
                person_feat_null = shape_feat

        audio_feat_in = [audio_feat_null]
        person_feat_in = [person_feat_null]
        for cond in cfg_cond:
            if cond == 'audio':
                audio_feat_in.append(audio_feat)
                person_feat_in.append(person_feat_null)
            elif cond == 'style':
                if cfg_mode == 'independent':
                    audio_feat_in.append(audio_feat_null)
                elif cfg_mode == 'incremental':
                    audio_feat_in.append(audio_feat)
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')
                person_feat_in.append(jt.concat([shape_feat, style_feat], dim=-1))

        n_entries = len(audio_feat_in)
        audio_feat_in = jt.concat(audio_feat_in, dim=0)
        person_feat_in = jt.concat(person_feat_in, dim=0)
        prev_motion_feat_in = jt.concat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = jt.concat([prev_audio_feat] * n_entries, dim=0)
        indicator_in = jt.concat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            if t > 1:
                z = jt.randn_like(motion_at_T)
            else:
                z = jt.zeros_like(motion_at_T)

            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = jt.concat([motion_at_t] * n_entries, dim=0)
            step_in = jt.array([t] * batch_size)
            step_in = jt.concat([step_in] * n_entries, dim=0)

            results = self.denoising_net(motion_in, audio_feat_in, person_feat_in, prev_motion_feat_in,
                                         prev_audio_feat_in, step_in, indicator_in)

            # Apply thresholding if specified
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                def quantile(input, q, dim):  
                    assert 0 <= q <= 1, "Quantile must be between 0 and 1"   
                    shape = input.shape  
                    sorted_input = jt.sort(input, dim=dim)[0]  
                    index = int(round(q * (shape[dim] - 1)))  
                    if index < 0:  
                        index = 0  
                    elif index >= shape[dim]:  
                        index = shape[dim] - 1    
                    if index < shape[dim] - 1:  
                        lower_bound = sorted_input[..., index]  
                        upper_bound = sorted_input[..., index + 1]   
                        weight = q * (shape[dim] - 1) - index  
          
                        quantile_values = lower_bound + weight * (upper_bound - lower_bound)  
                    else:  
                        quantile_values = sorted_input[..., index]  
                    return quantile_values 
                s = quantile(abs_results, dt_ratio, dim=1)
                s = jt.clamp(s, dt_min, dt_max)
                s = s[..., None, None]
                results = jt.minimum(jt.maximum(results, -s), s)

            results = results.chunk(n_entries)

            # Unconditional target (CFG) or the conditional target (non-CFG)
            target_theta = results[0][:, -self.n_motions:]
            # Classifier-free Guidance (optional)
            for i in range(0, n_entries - 1):
                if cfg_mode == 'independent':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (
                                results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:])
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

            if self.target == 'noise':
                c0 = 1 / jt.sqrt(alpha)
                c1 = (1 - alpha) / jt.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * jt.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * jt.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))

            traj[t - 1] = motion_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat
        else:
            return traj[0], motion_at_T, audio_feat


class DenoisingNetwork(nn.Module):
    def __init__(self, args, device='cuda'):
        super().__init__()

        # Model parameters
        self.use_style = args.style_enc_ckpt is not None
        self.motion_feat_dim = 50
        if args.rot_repr == 'aa':
            self.motion_feat_dim += 1 if args.no_head_pose else 4
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')
        self.shape_feat_dim = 100
        if self.use_style:
            self.style_feat_dim = args.d_style
            self.person_feat_dim = self.shape_feat_dim + self.style_feat_dim
        else:
            self.person_feat_dim = self.shape_feat_dim
        self.use_indicator = args.use_indicator

        # Transformer
        self.architecture = args.architecture
        self.feature_dim = args.feature_dim
        self.n_heads = args.n_heads
        self.n_layers = args.n_layers
        self.mlp_ratio = args.mlp_ratio
        self.align_mask_width = args.align_mask_width
        self.use_learnable_pe = not args.no_use_learnable_pe
        # sequence length
        self.n_prev_motions = args.n_prev_motions
        self.n_motions = args.n_motions

        # Temporal embedding for the diffusion time step
        self.TE = PositionalEncoding(self.feature_dim, max_len=args.n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)
        )

        if self.use_learnable_pe:
            # Learnable positional encoding
            self.PE = nn.Parameter(jt.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        self.person_proj = nn.Linear(self.person_feat_dim, self.feature_dim)

        # Transformer decoder
        if self.architecture == 'decoder':
            self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0),
                                          self.feature_dim)
            from mymodels import TransformerDecoderLayer
            from mymodels import TransformerDecoder
            decoder_layer_class = lambda: TransformerDecoderLayer(
                d_model=self.feature_dim, nhead=self.n_heads, dim_feedforward=self.mlp_ratio * self.feature_dim,
                activation='gelu', batch_first=True
            )
            self.transformer = TransformerDecoder(decoder_layer_class, num_layers=self.n_layers)
            if self.align_mask_width > 0:
                motion_len = self.n_prev_motions + self.n_motions
                alignment_mask = enc_dec_mask(motion_len, motion_len, 1, self.align_mask_width - 1)
                alignment_mask = jt.nn.pad(alignment_mask, (0, 0, 1, 0), value=0)
                self.alignment_mask = jt.array(alignment_mask)
            else:
                self.alignment_mask = None
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Motion decoder
        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim)
        )
        for param in self.parameters():
            param.requires_grad = True

    @property
    def device(self):
        return 'cuda' if jt.flags.use_cuda else 'cpu'

    def execute(self, motion_feat, audio_feat, person_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_motion). Noisy motion feature
            audio_feat: (N, L, feature_dim)
            person_feat: (N, 1, d_person)
            prev_motion_feat: (N, L_p, d_motion). Padded previous motion coefficients or feature
            prev_audio_feat: (N, L_p, d_audio). Padded previous motion coefficients or feature
            step: (N,)
            indicator: (N, L). 0/1 indicator for the real (unpadded) motion feature

        Returns:
            motion_feat_target: (N, L_p + L, d_motion)
        """
        # Diffusion time step embedding
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)  # (N, 1, diff_step_dim)

        person_feat = self.person_proj(person_feat)  # (N, 1, feature_dim)
        person_feat = person_feat + diff_step_embedding

        if indicator is not None:
            indicator = jt.concat([jt.zeros((indicator.shape[0], self.n_prev_motions)),
                                   indicator], dim=1)  # (N, L_p + L)
            indicator = indicator.unsqueeze(-1)  # (N, L_p + L, 1)

        # Concat features and embeddings
        if self.architecture == 'decoder':
            feats_in = jt.concat([prev_motion_feat, motion_feat], dim=1)  # (N, L_p + L, d_motion)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')
        if self.use_indicator:
            feats_in = jt.concat([feats_in, indicator], dim=-1)  # (N, L_p + L, d_motion + d_audio + 1)

        feats_in = self.feature_proj(feats_in)  # (N, L_p + L, feature_dim)
        feats_in = jt.concat([person_feat, feats_in], dim=1)  # (N, 1 + L_p + L, feature_dim)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        # Transformer
        if self.architecture == 'decoder':
            audio_feat_in = jt.concat([prev_audio_feat, audio_feat], dim=1)  # (N, L_p + L, d_audio)
            feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Decode predicted motion feature noise / sample
        motion_feat_target = self.motion_dec(feat_out[:, 1:])  # (N, L_p + L, d_motion)

        return motion_feat_target
