from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd
import os
from collections import deque, defaultdict
from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
import time

class SimpleOutputAdapter(nn.Module):
    def __init__(self, pred_len: int, buffer_context_size: int = 5, n_vars: int = 1, 
                 var_wise_gating: bool = False, num_layers: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.pred_len = pred_len
        self.buffer_context_size = buffer_context_size
        self.n_vars = n_vars
        self.var_wise = var_wise_gating
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        input_dim = self.pred_len + self.buffer_context_size
        output_dim = self.pred_len
        
        if self.var_wise:
            if self.num_layers == 1:
                self.fc_layers = nn.ModuleList([
                    nn.Linear(input_dim, output_dim) for _ in range(n_vars)
                ])
            else:
                self.fc_layers = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.Tanh(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_dim, output_dim)
                    ) for _ in range(n_vars)
                ])
            self.gate = nn.Parameter(torch.zeros(n_vars)) 
        else:
            if self.num_layers == 1:
                self.fc = nn.Linear(input_dim, output_dim)
            else:
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, output_dim)
                )
            self.gate = nn.Parameter(torch.zeros(1))
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        if self.var_wise:
            for fc_module in self.fc_layers:
                if self.num_layers == 1:
                    nn.init.xavier_uniform_(fc_module.weight, gain=0.1)
                    nn.init.zeros_(fc_module.bias)
                else:
                    for layer in fc_module:
                        if isinstance(layer, nn.Linear):
                            nn.init.xavier_uniform_(layer.weight, gain=0.1)
                            nn.init.zeros_(layer.bias)
        else:
            if self.num_layers == 1:
                nn.init.xavier_uniform_(self.fc.weight, gain=0.1)
                nn.init.zeros_(self.fc.bias)
            else:
                for layer in self.fc:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight, gain=0.1)
                        nn.init.zeros_(layer.bias)
        
    def forward(self, y: torch.Tensor, context_data: torch.Tensor = None):
        if context_data is None:
            return y
        
        batch_size, pred_len, n_vars = y.shape
        
        if self.var_wise:
            corrections = []
            for var_idx in range(n_vars):
                y_var = y[:, :, var_idx]
                combined_input = torch.cat([y_var, context_data], dim=-1)
                correction_var = self.fc_layers[var_idx](combined_input)
                corrections.append(correction_var.unsqueeze(-1))
            correction = torch.cat(corrections, dim=-1)
            gating_factor = torch.tanh(self.gate).unsqueeze(0).unsqueeze(0)
        else:
            y_flattened = y.transpose(1, 2).contiguous().view(batch_size * n_vars, pred_len)
            context_repeated = context_data.unsqueeze(1).repeat(1, n_vars, 1).view(batch_size * n_vars, -1)
            combined_input = torch.cat([y_flattened, context_repeated], dim=-1)
            correction = self.fc(combined_input)
            correction = correction.view(batch_size, n_vars, pred_len).transpose(1, 2)
            gating_factor = torch.tanh(self.gate)
        
        return y + gating_factor * correction


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
        self.per_batch_lr_reset = getattr(cfg.TTA.COSA, 'PER_BATCH_LR_RESET', True)
        self.loss_history = deque(maxlen=5)
        self.current_lr = getattr(cfg.TTA.SOLVER, 'BASE_LR', 0.001)
        self.sample_history = deque(maxlen=200)
        self.current_time_idx = 0
        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)

        self.output_adapter = SimpleOutputAdapter(
            pred_len=cfg.DATA.PRED_LEN,
            buffer_context_size=self.buffer_context_size,
            n_vars=cfg.DATA.N_VAR,
            var_wise_gating=self.var_wise_gating,
            num_layers=self.adapter_layers,
            hidden_dim=self.hidden_dim
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
        total_sum = 0
        for _, param in self.named_parameters():
            if param.requires_grad:
                total_sum += int(param.numel())
        return total_sum

    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError

    def reset(self):
        self._load_model_and_optimizer()
        self.adapters_enabled = False
        self.step_count = 0
        self.loss_history.clear()
        self.current_lr = getattr(self.cfg.TTA.SOLVER, 'BASE_LR', 0.001)

    def _copy_model_and_optimizer(self):
        return deepcopy(self.model.state_dict()), deepcopy(self.optimizer.state_dict())

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

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
        seq_mse = F.mse_loss(prediction, targets).item()
        self.sample_history.append({
            'time_idx': self.current_time_idx,
            'target_mean': targets.mean().item(),
        })
        self.current_time_idx += batch_size
        self.step_count += 1
        if not self.adapters_enabled:
            self.adapters_enabled = True
        return seq_mse

    def _get_individual_context_for_batch(self, batch_size, current_batch_idx):
        if len(self.sample_history) == 0:
            return torch.zeros(batch_size, self.buffer_context_size, device='cuda')
        history_size = min(self.buffer_context_size, len(self.sample_history))
        context_values = [self.sample_history[-(i+1)]['target_mean'] for i in range(history_size)]
        if len(context_values) < self.buffer_context_size:
            last_val = context_values[-1] if context_values else 0.0
            context_values.extend([last_val] * (self.buffer_context_size - len(context_values)))
        context_tensor = torch.tensor(context_values, dtype=torch.float32, device='cuda')
        return context_tensor.unsqueeze(0).expand(batch_size, -1)

    def _adaptive_learning_rate(self, current_loss: float, step: int, batch_idx: int = 0) -> float:
        if step == 0 and self.per_batch_lr_reset:
            self.current_lr = getattr(self.cfg.TTA.SOLVER, 'BASE_LR', 0.001)
            self.loss_history.append(current_loss)
            return self.current_lr
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

    @torch.enable_grad()
    def adapt_simple(self):
        from pathlib import Path
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        total_start_time = time.time()
        self.switch_model_to_eval()

        for _, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)

            while batch_end < len(enc_window_all):
                if self.paas_enabled:
                    enc_window_first = enc_window_all[batch_start]
                    _, batch_size, _ = self._calculate_period_and_batch_size(enc_window_first)
                else:
                    batch_size = getattr(self.cfg.TTA.COSA, 'BATCH_SIZE', 64)

                batch_end = min(batch_start + batch_size, len(enc_window_all))
                batch_size = batch_end - batch_start
                self.cur_step += batch_size

                batch_inputs = (
                    enc_window_all[batch_start:batch_end],
                    enc_window_stamp_all[batch_start:batch_end],
                    dec_window_all[batch_start:batch_end],
                    dec_window_stamp_all[batch_start:batch_end]
                )

                pred, ground_truth = forecast(self.cfg, batch_inputs, self.model, self.norm_module)
                original_pred = pred.clone()

                if self.adapters_enabled:
                    context_data = self._get_individual_context_for_batch(batch_size, batch_idx)
                    with torch.no_grad():
                        pred = self.output_adapter(pred, context_data)

                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)

                self.update_memory_buffer(ground_truth, original_pred)

                if self.adapters_enabled:
                    context_data = self._get_individual_context_for_batch(batch_size, batch_idx)
                    effective_steps = min(self.adapt_steps, 5) if self.fast_adaptation else self.adapt_steps

                    for step in range(effective_steps):
                        self.n_adapt += 1
                        self.switch_model_to_train()
                        adapted_pred = self.output_adapter(original_pred, context_data)
                        loss = F.mse_loss(adapted_pred, ground_truth)
                        if not hasattr(self, '_adapter_params'):
                            self._adapter_params = list(self.output_adapter.parameters())
                        l2_reg = sum(p.pow(2).sum() for p in self._adapter_params if p.requires_grad)
                        loss += 1e-4 * l2_reg

                        if self.adaptive_lr and self.fast_adaptation:
                            current_lr = self._adaptive_learning_rate(loss.item(), step, batch_idx)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = current_lr

                        self.optimizer.zero_grad()
                        loss.backward()
                        if self.fast_adaptation:
                            torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=max(0.05, min(0.5, loss.item())))
                        else:
                            torch.nn.utils.clip_grad_norm_(self.output_adapter.parameters(), max_norm=0.1)
                        self.optimizer.step()
                        self.switch_model_to_eval()

                        if (self.fast_adaptation and step > 2 and len(self.loss_history) >= 2 and
                                abs(self.loss_history[-1] - self.loss_history[-2]) < self.convergence_threshold):
                            break

                batch_start = batch_end
                batch_idx += 1

        self.time_stats['total_time'] = time.time() - total_start_time
        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)

        result_dir = Path(self.cfg.RESULT_DIR)
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "cosa_result.txt", "w") as f:
            f.write(
                f"adapt_test_mse: {float(self.mse_all.mean()):.6f}\n"
                f"adapt_test_mae: {float(self.mae_all.mean()):.6f}\n"
                f"n_adapt: {int(self.n_adapt)}\n"
            )
        print(
            f"[COSA] adapt_test_mse={float(self.mse_all.mean()):.4f}, "
            f"adapt_test_mae={float(self.mae_all.mean()):.4f}, "
            f"n_adapt={int(self.n_adapt)}",
            flush=True
        )

    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
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

    def adapt(self):
        self.adapt_simple()


def build_adapter(cfg, model, norm_module=None):
    adapter = SimpleAdapter(cfg, model, norm_module)
    return adapter
