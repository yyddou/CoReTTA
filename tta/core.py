from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
import os
from collections import deque, defaultdict
import numpy as np
from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
import time

class SimpleOutputAdapter(nn.Module):
    def __init__(
        self,
        pred_len: int,
        buffer_context_size: int = 5,
        n_vars: int = 1,
        var_wise_gating: bool = False,
        num_layers: int = 1,
        hidden_dim: int = 64,
        gate_bias_init: float = -1.0,
        scr_rank_mult: float = 1.0,
    ):
        super().__init__()
        self.pred_len          = pred_len
        self.buffer_context_size = buffer_context_size
        self.n_vars            = n_vars
        self.var_wise          = var_wise_gating
        self.num_layers        = num_layers
        self.hidden_dim        = hidden_dim

        # -- Base adapter: time-domain branch --------------------
        input_dim  = pred_len + buffer_context_size
        output_dim = pred_len

        if self.var_wise:
            if num_layers == 1:
                self.fc_layers = nn.ModuleList([
                    nn.Linear(input_dim, output_dim) for _ in range(n_vars)
                ])
            else:
                self.fc_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.Tanh(), nn.Dropout(0.1),
                        nn.Linear(hidden_dim, output_dim),
                    ) for _ in range(n_vars)
                ])
            self.gate = nn.Parameter(torch.zeros(n_vars))
        else:
            if num_layers == 1:
                self.fc = nn.Linear(input_dim, output_dim)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(), nn.Dropout(0.1),
                    nn.Linear(hidden_dim, output_dim),
                )
            self.gate = nn.Parameter(torch.zeros(1))

        self._initialize_parameters()

        # -- SCR: cross-variate correction refinement -------------
        _r = max(1, int(round(n_vars * scr_rank_mult)))  # rank r = C * scr_rank_mult
        self.scr_down = nn.Linear(2 * pred_len, _r)
        self.scr_up   = nn.Linear(_r, pred_len)
        nn.init.xavier_uniform_(self.scr_down.weight, gain=0.01)
        nn.init.zeros_(self.scr_down.bias)
        nn.init.xavier_uniform_(self.scr_up.weight, gain=0.01)
        nn.init.zeros_(self.scr_up.bias)
        # SCR gate: per-variate, conditioned on spectral descriptors
        self.scr_gate_net = nn.Linear(4, n_vars)
        nn.init.zeros_(self.scr_gate_net.weight)
        self.gate_bias_init = gate_bias_init
        nn.init.constant_(self.scr_gate_net.bias, gate_bias_init)

    def _initialize_parameters(self):
        modules = self.fc_layers if self.var_wise else [self.fc]
        for m in modules:
            layers = [m] if isinstance(m, nn.Linear) else m
            for layer in layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.1)
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        y: torch.Tensor,                   # [B, pred_len, n_vars]
        context_data: torch.Tensor = None, # [B, buffer_ctx]
        spectral_ctx: torch.Tensor = None, # [B, 4] for STAR gate only
    ) -> torch.Tensor:

        B, T, V = y.shape

        # -- Base adapter correction ---------------------------
        if context_data is not None:
            if self.var_wise:
                corrections = []
                for i in range(V):
                    combined = torch.cat([y[:, :, i], context_data], dim=-1)
                    corrections.append(self.fc_layers[i](combined).unsqueeze(-1))
                correction_td = torch.cat(corrections, dim=-1)          # [B, T, V]
                gate_td = torch.tanh(self.gate).unsqueeze(0).unsqueeze(0)
            else:
                y_flat  = y.transpose(1, 2).contiguous().view(B * V, T)
                ctx_rep = context_data.unsqueeze(1).repeat(1, V, 1).view(B * V, -1)
                correction_td = (
                    self.fc(torch.cat([y_flat, ctx_rep], dim=-1))
                    .view(B, V, T).transpose(1, 2)
                )                                                        # [B, T, V]
                gate_td = torch.tanh(self.gate)                          # scalar
        else:
            correction_td = torch.zeros_like(y)
            gate_td       = 0.0
        out = y + gate_td * correction_td

        # -- SCR: cross-variate correction refinement -------------
        # delta = adapter correction so far; anchor = mean over variates
        # each variate: [delta_c || anchor] -> low-rank bottleneck -> ref_c (Eq. bottleneck)
        correction = out - y                           # [B, T, V]

        corr_T     = correction.permute(0, 2, 1)       # [B, V, T]
        core       = corr_T.mean(dim=1, keepdim=True)  # [B, 1, T]
        core_e     = core.expand_as(corr_T)            # [B, V, T]
        delta_cross = self.scr_up(
            torch.tanh(self.scr_down(
                torch.cat([corr_T, core_e], dim=-1)    # [B, V, 2T]
            ))                                         # [B, V, r]
        ).permute(0, 2, 1)                             # [B, T, V]
        # per-variate spectral gate g_t (Eq. gate)
        if spectral_ctx is not None:
            freq_ctx_g = spectral_ctx  # [B, 4]
        else:
            freq_ctx_g = torch.zeros(B, 4, device=out.device)
        gate = torch.tanh(self.scr_gate_net(freq_ctx_g)).unsqueeze(1)  # [B, 1, V]

        out = out + gate * delta_cross

        return out


# -------------------------------------------------------------
#  CoReAdapter
# -------------------------------------------------------------
class SimpleAdapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(SimpleAdapter, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        self.buffer_context_size = getattr(cfg.TTA.COSA, 'BUFFER_CONTEXT_SIZE', 5)
        self.adapt_steps = getattr(cfg.TTA.COSA, 'STEPS', 20)
        
        self.paas_enabled = getattr(cfg.TTA.COSA, 'PAAS', False)
        self.period_n = getattr(cfg.TTA.COSA, 'PERIOD_N', 1)
        
        self.fast_adaptation = getattr(cfg.TTA.COSA, 'FAST_ADAPTATION', False)
        self.adaptive_lr = getattr(cfg.TTA.COSA, 'ADAPTIVE_LR', False) 
        self.max_lr = getattr(cfg.TTA.COSA, 'MAX_LR', 0.005)
        self.min_lr = getattr(cfg.TTA.COSA, 'MIN_LR', 0.0001)
        self.convergence_threshold = getattr(cfg.TTA.COSA, 'CONVERGENCE_THRESHOLD', 1e-4)
        self.var_wise_gating = getattr(cfg.TTA.COSA, 'VAR_WISE_GATING', False)
        
        self.save_csv = getattr(cfg.TTA.COSA, 'SAVE_CSV', False)
        self.csv_predictions = [] 

        self.save_paas_csv = getattr(cfg.TTA.COSA, 'SAVE_PAAS_CSV', False)
        self.paas_batch_info = []

        self.adapter_layers = getattr(cfg.TTA.COSA, 'ADAPTER_LAYERS', 1)
        self.hidden_dim = getattr(cfg.TTA.COSA, 'HIDDEN_DIM', 64)

        # Per-Batch LR Reset
        self.per_batch_lr_reset = getattr(cfg.TTA.COSA, 'PER_BATCH_LR_RESET', True)

        self.loss_history = deque(maxlen=5)
        self.current_lr = getattr(cfg.TTA.SOLVER, 'BASE_LR', 0.001)
        self.lr_freq_scale     = float(getattr(cfg.TTA.COSA, 'LR_FREQ_SCALE', 0.5))
        self._cur_spec_entropy = 0.0
        self._cur_low_band     = 0.0
        self._cur_mid_band     = 0.0
        self._cur_centroid     = 0.0
        self._ref_low_band     = None
        self._ref_mid_band     = None
        self._ref_centroid     = None
        self._ref_alpha        = 0.05 

        self.sample_history = deque(maxlen=200) 
        self.current_time_idx = 0
        
        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)
        
        gate_bias_init = getattr(cfg.TTA.CORE, 'GATE_BIAS_INIT', -1.0)
        scr_rank_mult = getattr(cfg.TTA.CORE, 'SCR_RANK_MULT', 1.0)
        self.output_adapter = SimpleOutputAdapter(
            pred_len=cfg.DATA.PRED_LEN,
            buffer_context_size=self.buffer_context_size,
            n_vars=cfg.DATA.N_VAR,
            var_wise_gating=self.var_wise_gating,
            num_layers=self.adapter_layers,
            hidden_dim=self.hidden_dim,
            gate_bias_init=gate_bias_init,
            scr_rank_mult=scr_rank_mult,
        ).cuda()
        
        self.adapters_enabled = False
        self.step_count = 0
        
        self._freeze_all_model_params()
        self._unfreeze_adapter_params()
        
        self.optimizer = get_optimizer(self.output_adapter.parameters(), cfg.TTA)
        
        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []

    def count_parameters(self):
        trainable_params = []
        total_sum = 0
        
        for name, param in self.named_parameters():
            param_info = {
                "name": name,
                "requires_grad": param.requires_grad,
                "size": list(param.size()),
                "numel": int(param.numel())
            }
            trainable_params.append(param_info)
            
            if param.requires_grad:
                total_sum += int(param.numel())
        
        param_json = {
            "model": "CoRe",
            "parameters": {
                "trainable_params": trainable_params,
                "total_params": total_sum
            }
        }    
        
    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError
    
    def reset(self):
        self._load_model_and_optimizer()
        self.adapters_enabled = False
        self.step_count = 0
        
        self.loss_history.clear()
        self.current_lr = getattr(self.cfg.TTA.SOLVER, 'BASE_LR', 0.001)
    
    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
    
    def _get_all_models(self):
        models = [self.model, self.output_adapter]
        if self.norm_module is not None:
            models.append(self.norm_module)
        return models

    def _freeze_all_model_params(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        if self.norm_module is not None:
            for param in self.norm_module.parameters():
                param.requires_grad_(False)
    
    def _unfreeze_adapter_params(self):
        for param in self.output_adapter.parameters():
            param.requires_grad_(True)
    
    def switch_model_to_train(self):
        self.model.eval() 
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.train()
    
    def switch_model_to_eval(self):
        self.model.eval()
        if self.norm_module is not None:
            self.norm_module.eval()
        self.output_adapter.eval()
    

    def update_memory_buffer(self, targets: torch.Tensor, prediction: torch.Tensor):

        batch_size = targets.shape[0]
        
        # Store only batch-level statistics (omit individual sample info for speed improvement)
        seq_mse = F.mse_loss(prediction, targets).item()
        
        # Store only batch-wide average values
        batch_info = {
            'time_idx': self.current_time_idx,
            'target_mean': targets.mean().item(),
        }
        self.sample_history.append(batch_info)
        
        self.current_time_idx += batch_size
        self.step_count += 1
        
        if not self.adapters_enabled:
            self.adapters_enabled = True
        
        return seq_mse
    
    def _calculate_period_and_batch_size(self, enc_window_first):

        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        
        dominant_freq = None
        try:
            dominant_freq_idx = torch.argmax(amplitude[:, power.argmax()]).item()
            dominant_freq = float(dominant_freq_idx / enc_window_first.shape[0])
            period = enc_window_first.shape[0] // dominant_freq_idx
        except:
            period = 24 
            dominant_freq = 1.0 / 24 
            
        period *= self.period_n
        batch_size = period + 1
        return period, batch_size, dominant_freq

    def _get_individual_context_for_batch(self, batch_size, current_batch_idx):

        if len(self.sample_history) == 0:
            return torch.zeros(batch_size, self.buffer_context_size, device='cuda')
        
        history_size = min(self.buffer_context_size, len(self.sample_history))
        context_values = [self.sample_history[-(i+1)]['target_mean'] 
                         for i in range(history_size)]
        
        if len(context_values) < self.buffer_context_size:
            last_val = context_values[-1] if context_values else 0.0
            context_values.extend([last_val] * (self.buffer_context_size - len(context_values)))
        
        context_tensor = torch.tensor(context_values, dtype=torch.float32, device='cuda')
        return context_tensor.unsqueeze(0).expand(batch_size, -1) 
    
    def _compute_spectral_context(self, enc_window: torch.Tensor):
        eps  = 1e-8
        x    = enc_window - enc_window.mean(dim=1, keepdim=True)
        fft  = torch.fft.rfft(x, dim=1)
        pow_ = (fft.real**2 + fft.imag**2).mean(dim=2).mean(dim=0)
        K    = pow_.shape[0]
        pi   = (pow_ + eps) / (pow_.sum() + eps * K)
        H    = float(-(pi * torch.log(pi + eps)).sum() / np.log(max(K, 2)))
        freqs = torch.linspace(0, 1, K, device=pow_.device)
        total = pow_.sum() + eps
        LBR   = float(pow_[:max(1, K//3)].sum() / total)
        MBR   = float(pow_[max(1, K//3):max(1, 2*K//3)].sum() / total)
        HBR   = float(pow_[max(1, 2*K//3):].sum() / total)
        C     = float((freqs * pow_).sum() / total)
        self._cur_spec_entropy = H
        self._cur_low_band     = LBR
        self._cur_mid_band     = MBR
        self._cur_high_band    = HBR
        self._cur_centroid     = C
        if self._ref_low_band is None:
            self._ref_low_band = LBR
            self._ref_mid_band = MBR
            self._ref_centroid = C
        else:
            a = self._ref_alpha
            self._ref_low_band = (1-a)*self._ref_low_band + a*LBR
            self._ref_mid_band = (1-a)*self._ref_mid_band + a*MBR
            self._ref_centroid = (1-a)*self._ref_centroid + a*C
        # store mid/high for reference
        if not hasattr(self, '_cur_mid_band'):
            self._cur_mid_band  = MBR
            self._cur_high_band = HBR

    def _adaptive_learning_rate(self, current_loss: float, step: int, batch_idx: int = 0) -> float:
        if step == 0 and self.per_batch_lr_reset:
            self.current_lr = getattr(self.cfg.TTA.SOLVER, 'BASE_LR', 0.001)
        base_lr = getattr(self.cfg.TTA.SOLVER, 'BASE_LR', 0.001)
        if self.lr_freq_scale > 0 and self._ref_low_band is not None:
            sev = float(np.clip(
                0.5  * self._cur_spec_entropy
                + 0.25 * abs(self._cur_low_band - self._ref_low_band)
                + 0.25 * abs(self._cur_mid_band - self._ref_mid_band),
                0.0, 1.0,
            ))
            return float(np.clip(
                base_lr * (1.0 + self.lr_freq_scale * sev),
                self.min_lr, self.max_lr,
            ))
        return base_lr

        self.loss_history.append(current_loss)
        recent_losses = list(self.loss_history)[-3:]

        if len(recent_losses) < 2:
            return self.current_lr

        loss_trend = recent_losses[-1] - recent_losses[0]
        loss_variance = torch.tensor(recent_losses).var().item()

        if loss_trend > 0 and loss_variance < 1e-6:
            self.current_lr = min(self.current_lr * 1.2, self.max_lr)
        elif loss_trend < -0.01:
            self.current_lr = min(self.current_lr * 1.05, self.max_lr)
        elif abs(loss_trend) < 1e-6:
            self.current_lr = max(self.current_lr * 0.8, self.min_lr)

        if step >= 1:
            cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(step * 3.14159 / self.adapt_steps)))
            self.current_lr = self.min_lr + (self.current_lr - self.min_lr) * cosine_factor

        return self.current_lr
    
    def _save_batch_predictions_to_csv(self, original_pred: torch.Tensor, tta_pred: torch.Tensor, ground_truth: torch.Tensor, batch_idx: int):
        batch_size, pred_len, n_vars = tta_pred.shape
        
        original_pred_np = original_pred.detach().cpu().numpy()
        tta_pred_np = tta_pred.detach().cpu().numpy()
        ground_truth_np = ground_truth.detach().cpu().numpy()
        
        for sample_idx in range(batch_size):
            for time_idx in range(pred_len):
                for var_idx in range(n_vars):
                    orig_val = float(original_pred_np[sample_idx, time_idx, var_idx])
                    tta_val = float(tta_pred_np[sample_idx, time_idx, var_idx])
                    gt_val = float(ground_truth_np[sample_idx, time_idx, var_idx])
                    
                    row_data = {
                        'batch_idx': batch_idx,
                        'sample_idx': sample_idx,
                        'global_sample_idx': batch_idx * batch_size + sample_idx,
                        'timestep': time_idx + 1, 
                        'variable_idx': var_idx,
                        'original_prediction': orig_val,
                        'tta_prediction': tta_val,
                        'ground_truth': gt_val,
                        'tta_improvement': tta_val - orig_val,
                        'original_absolute_error': float(abs(orig_val - gt_val)),
                        'tta_absolute_error': float(abs(tta_val - gt_val)),
                        'original_squared_error': float((orig_val - gt_val)**2),
                        'tta_squared_error': float((tta_val - gt_val)**2),
                        'error_improvement': float(abs(orig_val - gt_val) - abs(tta_val - gt_val)) 
                    }
                    self.csv_predictions.append(row_data)
    
    def _export_predictions_to_csv(self):
        """Export stored predictions to CSV file"""
        if not self.csv_predictions:
            return
        
        df = pd.DataFrame(self.csv_predictions)
        
        model_name = self.cfg.MODEL.NAME
        dataset_name = self.cfg.DATA.NAME
        pred_len = self.cfg.DATA.PRED_LEN
        
        csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "CoRe")
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_filename = f"{model_name}_{dataset_name}_pred{pred_len}_predictions.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        df.to_csv(csv_path, index=False)
    
    def _save_paas_batch_info(self, batch_idx: int, batch_start: int, calculated_batch_size: int, 
                             actual_batch_size: int, period: int = None, fft_dominant_freq: float = None):
        batch_info = {
            'batch_idx': batch_idx,
            'batch_start_idx': batch_start,
            'calculated_batch_size': calculated_batch_size,
            'actual_batch_size': actual_batch_size,
            'period': period if period is not None else 'N/A',
            'fft_dominant_frequency': fft_dominant_freq if fft_dominant_freq is not None else 'N/A',
            'paas_enabled': self.paas_enabled,
            'period_n_multiplier': self.period_n
        }
        self.paas_batch_info.append(batch_info)
    
    def _export_paas_info_to_csv(self):
        if not self.paas_batch_info:
            return
        
        df = pd.DataFrame(self.paas_batch_info)
        
        model_name = self.cfg.MODEL.NAME
        dataset_name = self.cfg.DATA.NAME
        pred_len = self.cfg.DATA.PRED_LEN
        
        csv_dir = os.path.join(self.cfg.RESULT_DIR, "csv_predictions", "CoRe")
        os.makedirs(csv_dir, exist_ok=True)
        
        csv_filename = f"{model_name}_{dataset_name}_pred{pred_len}_paas_batch_sizes.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        df.to_csv(csv_path, index=False)
    
    @torch.enable_grad()
    def adapt_simple(self):
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        
        total_start_time = time.time()
        
        self.switch_model_to_eval()
        
        for _, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            
            while batch_end < len(enc_window_all):
                calculated_batch_size = None
                period = None
                dominant_freq = None
                
                if self.paas_enabled:
                    period_start = time.time()
                    enc_window_first = enc_window_all[batch_start]
                    period, calculated_batch_size, dominant_freq = self._calculate_period_and_batch_size(enc_window_first)
                    batch_size = calculated_batch_size
                    self.time_stats['paas_period_calculation'] += time.time() - period_start
                    self.time_counts['paas_period_calculation'] += 1
                else:
                    batch_size = getattr(self.cfg.TTA.COSA, 'BATCH_SIZE', 64)
                    calculated_batch_size = batch_size
                
                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start
                
                if self.save_paas_csv:
                    self._save_paas_batch_info(
                        batch_idx=batch_idx,
                        batch_start=batch_start,
                        calculated_batch_size=calculated_batch_size,
                        actual_batch_size=batch_size,
                        period=period,
                        fft_dominant_freq=dominant_freq
                    )

                self.cur_step += batch_size

                batch_inputs = (
                    enc_window_all[batch_start:batch_end], 
                    enc_window_stamp_all[batch_start:batch_end], 
                    dec_window_all[batch_start:batch_end], 
                    dec_window_stamp_all[batch_start:batch_end]
                )
                
                self._compute_spectral_context(enc_window_all[batch_start:batch_end])
                pred_start = time.time()
                pred, ground_truth = forecast(self.cfg, batch_inputs, self.model, self.norm_module)
                original_pred = pred.clone()
                self.time_stats['base_prediction'] += time.time() - pred_start
                self.time_counts['base_prediction'] += 1

                if self.adapters_enabled:
                    context_start = time.time()
                    context_data = self._get_individual_context_for_batch(batch_size, batch_idx)
                    self.time_stats['context_generation'] += time.time() - context_start
                    self.time_counts['context_generation'] += 1

                    final_start = time.time()
                    spec_ctx = torch.tensor(
                        [[self._cur_spec_entropy, self._cur_low_band, self._cur_mid_band, self._cur_high_band]],
                        dtype=torch.float32, device='cuda'
                    ).expand(batch_size, -1)
                    with torch.no_grad():
                        pred = self.output_adapter(pred, context_data,
                                                   spectral_ctx=spec_ctx)
                    self.time_stats['final_prediction'] += time.time() - final_start
                    self.time_counts['final_prediction'] += 1

                if self.save_csv:
                    self._save_batch_predictions_to_csv(original_pred, pred, ground_truth, batch_idx)

                metric_start = time.time()
                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.time_stats['metric_computation'] += time.time() - metric_start
                self.time_counts['metric_computation'] += 1

                self.mse_all.append(mse)
                self.mae_all.append(mae)

                buffer_start = time.time()
                self.update_memory_buffer(ground_truth, original_pred)
                self.time_stats['buffer_update'] += time.time() - buffer_start
                self.time_counts['buffer_update'] += 1

                if self.adapters_enabled:
                    context_data = self._get_individual_context_for_batch(batch_size, batch_idx)

                    effective_steps = self.adapt_steps
                    if self.fast_adaptation:
                        effective_steps = min(self.adapt_steps, 5)

                    # ensure spec_ctx is defined even when adapters_enabled was False
                    if 'spec_ctx' not in dir():
                        spec_ctx = torch.tensor(
                            [[self._cur_spec_entropy, self._cur_low_band, self._cur_mid_band, self._cur_high_band]],
                            dtype=torch.float32, device='cuda'
                        ).expand(batch_size, -1)
                    adapt_start = time.time()
                    for step in range(effective_steps):
                        step_start = time.time()

                        self.n_adapt += 1
                        self.switch_model_to_train()

                        forward_start = time.time()
                        adapted_pred = self.output_adapter(original_pred, context_data,
                                                              spectral_ctx=spec_ctx)
                        self.time_stats['adapter_forward'] += time.time() - forward_start
                        self.time_counts['adapter_forward'] += 1

                        loss_start = time.time()
                        loss = F.mse_loss(adapted_pred, ground_truth)

                        if not hasattr(self, '_adapter_params'):
                            self._adapter_params = list(self.output_adapter.parameters())
                        l2_reg = sum(p.pow(2).sum() for p in self._adapter_params if p.requires_grad)
                        loss += 1e-4 * l2_reg

                        self.time_stats['loss_computation'] += time.time() - loss_start
                        self.time_counts['loss_computation'] += 1

                        if self.adaptive_lr and self.fast_adaptation:
                            current_lr = self._adaptive_learning_rate(loss.item(), step, batch_idx)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = current_lr

                        backward_start = time.time()
                        self.optimizer.zero_grad()
                        loss.backward()

                        if self.fast_adaptation:
                            max_norm = max(0.05, min(0.5, loss.item()))
                            torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=max_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=0.1)

                        self.optimizer.step()
                        self.time_stats['backward_update'] += time.time() - backward_start
                        self.time_counts['backward_update'] += 1

                        self.switch_model_to_eval()

                        if (self.fast_adaptation and step > 2 and len(self.loss_history) >= 2 and
                            abs(self.loss_history[-1] - self.loss_history[-2]) < self.convergence_threshold):
                            break

                        self.time_stats['per_adaptation_step'] += time.time() - step_start
                        self.time_counts['per_adaptation_step'] += 1

                    self.time_stats['total_adaptation'] += time.time() - adapt_start
                    self.time_counts['total_adaptation'] += 1
                
                batch_start = batch_end
                batch_idx += 1
        
        self.time_stats['total_time'] = time.time() - total_start_time
        
        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1
        
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        import pandas as pd
        pd.DataFrame({'batch_mse': self.mse_all}).to_csv(
            f"{self.cfg.RESULT_DIR}/per_batch_mse.csv", index=False)

        from pathlib import Path
        result_dir = Path(self.cfg.RESULT_DIR)
        result_dir.mkdir(parents=True, exist_ok=True)

        with open(result_dir / "core_result.txt", "w") as f:
            f.write(
                f"adapt_test_mse: {float(self.mse_all.mean()):.6f}\n"
                f"adapt_test_mae: {float(self.mae_all.mean()):.6f}\n"
                f"n_adapt: {int(self.n_adapt)}\n"
            )

        print(
            f"[CoRe] result_dir={result_dir} "
            f"adapt_test_mse={float(self.mse_all.mean()):.6f} "
            f"adapt_test_mae={float(self.mae_all.mean()):.6f} "
            f"n_adapt={int(self.n_adapt)}",
            flush=True
        )





        from pathlib import Path
        result_dir = Path(self.cfg.RESULT_DIR)
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "core_result.txt", "w") as f:
            f.write(
                f"adapt_test_mse: {float(self.mse_all.mean()):.6f}\n"
                f"adapt_test_mae: {float(self.mae_all.mean()):.6f}\n"
                f"n_adapt: {int(self.n_adapt)}\n"
            )

        print(
            f"[CoRe] adapt_test_mse={float(self.mse_all.mean()):.4f}, "
            f"adapt_test_mae={float(self.mae_all.mean()):.4f}, "
            f"n_adapt={int(self.n_adapt)}",
            flush=True
        )
        assert len(self.mse_all) == len(self.test_loader.dataset)
        
        if self.save_csv:
            self._export_predictions_to_csv()
        
        if self.save_paas_csv:
            self._export_paas_info_to_csv()

        self._print_combined_results()
        
    
    def _print_combined_results(self):
        total_params = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                total_params += int(param.numel())
        
        avg_times = {}
        for key in self.time_stats:
            if key != 'total_time':
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count
        
        adapter_total = (avg_times.get('context_generation', 0) + 
                        avg_times.get('adapter_forward', 0) + 
                        avg_times.get('final_prediction', 0))
        
        time_statistics = {
            "adapter_operations": {},
            "adaptation_training": {},
            "other_operations": {},
            "overall_stats": {}
        }
        
        time_statistics["adapter_operations"] = {
            "context_generation_ms": round(avg_times.get('context_generation', 0) * 1000, 3),
            "adapter_forward_pass_ms": round(avg_times.get('adapter_forward', 0) * 1000, 3),
            "final_prediction_ms": round(avg_times.get('final_prediction', 0) * 1000, 3),
            "total_adapter_operation_ms": round(adapter_total * 1000, 3)
        }
        
        time_statistics["adaptation_training"] = {
            "loss_computation_ms": round(avg_times.get('loss_computation', 0) * 1000, 3),
            "backward_update_ms": round(avg_times.get('backward_update', 0) * 1000, 3),
            "per_adaptation_step_ms": round(avg_times.get('per_adaptation_step', 0) * 1000, 3),
            "total_per_batch_ms": round(avg_times.get('total_adaptation', 0) * 1000, 3),
            "adaptation_steps": self.adapt_steps
        }
        
        other_ops = {
            "base_model_prediction_ms": round(avg_times.get('base_prediction', 0) * 1000, 3),
            "buffer_update_ms": round(avg_times.get('buffer_update', 0) * 1000, 3),
            "metric_computation_ms": round(avg_times.get('metric_computation', 0) * 1000, 3)
        }
        if self.paas_enabled:
            other_ops["paas_period_calculation_ms"] = round(avg_times.get('paas_period_calculation', 0) * 1000, 3)
        time_statistics["other_operations"] = other_ops
        
        adaptation_count = max(self.time_counts.get('total_adaptation', 1), 1)
        
        time_statistics["overall_stats"] = {
            "total_time_seconds": round(self.time_stats['total_time'], 2),
            "total_adaptations": int(self.n_adapt),
            "avg_time_per_adaptation_ms": round(self.time_stats.get('total_adaptation', 0) / adaptation_count * 1000, 3),
            "throughput_samples_per_sec": round(len(self.test_loader.dataset) / self.time_stats['total_time'], 1)
        }
        
        combined_results = {
            "model": "CoRe",
            "time_statistics": time_statistics,
            "final_results": {
                "adaptation_count": int(self.n_adapt),
                "test_mse": float(self.mse_all.mean())
            },
            "parameters": {
                "total_params": total_params
            }
        }
        
        from pathlib import Path
        result_dir = Path(self.cfg.RESULT_DIR)
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "core_results.json", "w") as f:
            json.dump(combined_results, f, indent=2)

        print(json.dumps(combined_results, indent=2))
    def adapt(self):
        self.adapt_simple()


def build_adapter(cfg, model, norm_module=None):
    adapter = SimpleAdapter(cfg, model, norm_module)
    return adapter