from typing import List
from collections import deque, defaultdict
from copy import deepcopy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import time

from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method
import matplotlib.pyplot as plt


class DynaTTAAdapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(DynaTTAAdapter, self).__init__()
        self.cfg = cfg
        self.model = model
        self.norm_module = norm_module
        self.norm_method = get_norm_method(cfg)

        cfg.TTA.TAFAS.CALI_MODULE = True
        if cfg.TTA.TAFAS.CALI_MODULE:
            self.cali = Calibration(cfg).to(next(self.model.parameters()).device)

        self._freeze_all_model_params()
        self.named_modules_to_adapt = self._get_named_modules_to_adapt()
        self._unfreeze_modules_to_adapt()
        self.named_params_to_adapt = self._get_named_params_to_adapt()
        self.optimizer = get_optimizer(self.named_params_to_adapt.values(), cfg.TTA)

        self.model_state, self.opt_state = self._copy_state()

        cfg.TEST.BATCH_SIZE = len(get_test_dataloader(cfg).dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_end = {}
        self.inputs_hist = {}
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []

        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)

        dyn = cfg.TTA.DYNATTA
        self.mse_buffer = deque(maxlen=dyn.MSE_BUFFER_SIZE)
        self.rtab = {}
        self.rdb = {}
        self.metric_hist = [deque(maxlen=dyn.METRIC_HISTORY_SIZE) for _ in range(3)]
        self.alpha_t = dyn.ALPHA_MIN
        self.alpha_min = dyn.ALPHA_MIN
        self.alpha_max = dyn.ALPHA_MAX
        self.kappa = dyn.KAPPA
        self.eta = dyn.ETA
        self.eps = dyn.EPS
        self.warmup_steps = dyn.WARMUP_FACTOR * cfg.DATA.PRED_LEN
        self.lr_history = []

        self.buffer_interval = dyn.UPDATE_BUFFERS_INTERVAL
        self.metric_interval = dyn.UPDATE_METRICS_INTERVAL

        self.steps_since_last_buffer_update = self.steps_since_last_metric_update = 0

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
            "model": "DynaTTAAdapter",
            "parameters": {
                "trainable_params": trainable_params,
                "total_params": total_sum
            }
        }


    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _freeze_all_model_params(self):
        for model in self._get_all_models():
            for param in model.parameters():
                param.requires_grad_(False)

    def _get_all_models(self):
        models = [self.model]
        if self.norm_module is not None:
            models.append(self.norm_module)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            models.append(self.cali)
        return models

    def _get_named_modules(self):
        named_modules = []
        for model in self._get_all_models():
            named_modules += list(model.named_modules())
        return named_modules

    def _get_named_modules_to_adapt(self) -> List[str]:
        named_modules = self._get_named_modules()
        if self.cfg.TTA.MODULE_NAMES_TO_ADAPT == 'all':
            return named_modules

        named_modules_to_adapt = []
        for module_name in self.cfg.TTA.MODULE_NAMES_TO_ADAPT.split(','):
            exact_match = '(exact)' in module_name
            module_name = module_name.replace('(exact)', '')
            if exact_match:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if name == module_name]
            else:
                named_modules_to_adapt += [(name, module) for name, module in named_modules if module_name in name]

        if len(named_modules_to_adapt) == 0:
            print(f"Warning: No modules found to adapt. Available modules:")
            for name, module in named_modules[:10]:
                print(f"  {name}: {type(module)}")
            print("Requested modules:", self.cfg.TTA.MODULE_NAMES_TO_ADAPT.split(','))

            calibration_modules = [(name, module) for name, module in named_modules
                                 if 'cali' in name.lower() or 'calibration' in name.lower()]

            if calibration_modules:
                print("Found calibration modules, using those for adaptation.")
                named_modules_to_adapt = calibration_modules
            else:
                norm_linear_modules = [(name, module) for name, module in named_modules
                                     if any(layer_type in str(type(module)).lower()
                                           for layer_type in ['norm', 'linear', 'embedding'])]
                if norm_linear_modules:
                    print("Using normalization/linear layers for adaptation.")
                    named_modules_to_adapt = norm_linear_modules[:5]
                else:
                    print("Falling back to all modules for adaptation.")
                    named_modules_to_adapt = named_modules

        return named_modules_to_adapt

    def _get_named_params_to_adapt(self):
        named_params_to_adapt = {}
        for model in self._get_all_models():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    named_params_to_adapt[name] = param
        return named_params_to_adapt

    def reset(self):
        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.opt_state)

    def adapt(self):
        return self.adapt_tafas()

    @torch.enable_grad()
    def adapt_tafas(self):
        self.switch_eval()
        batch_start, batch_end, batch_idx = 0, 0, 0

        total_start_time = time.time()

        for _, inputs in enumerate(self.test_loader):
            enc_all, enc_stamp, dec_all, dec_stamp = prepare_inputs(inputs)
            total = enc_all.shape[0]

            while batch_end < total:
                enc0 = enc_all[batch_start]
                if self.cfg.TTA.TAFAS.PAAS:
                    period, bs = self._calc_period(enc0)
                else:
                    bs = self.cfg.TTA.TAFAS.BATCH_SIZE
                    period = bs - 1
                batch_end = min(batch_start + bs, total)
                bs = batch_end - batch_start
                self.cur_step += bs
                self.steps_since_last_buffer_update += bs
                self.steps_since_last_metric_update += bs

                window = (
                    enc_all[batch_start:batch_end],
                    enc_stamp[batch_start:batch_end],
                    dec_all[batch_start:batch_end],
                    dec_stamp[batch_start:batch_end]
                )
                self.pred_end[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_hist[batch_idx] = window

                full_adapt_start = time.time()
                self._adapt_full()
                self.time_stats['full_adaptation'] += time.time() - full_adapt_start
                self.time_counts['full_adaptation'] += 1

                partial_adapt_start = time.time()
                pred, gt = self._adapt_partial(window, period, bs, batch_idx)
                self.time_stats['partial_adaptation'] += time.time() - partial_adapt_start
                self.time_counts['partial_adaptation'] += 1

                metrics_start = time.time()
                z, dr, dp = self._collect_current_metrics(window)
                device = next(self.model.parameters()).device
                metrics = torch.tensor([z, dr, dp], device=device)
                self.time_stats['metrics_collection'] += time.time() - metrics_start
                self.time_counts['metrics_collection'] += 1

                if self.cfg.TTA.TAFAS.ADJUST_PRED:
                    adjust_start = time.time()
                    pred, gt = self._adjust_prediction(pred, window, bs, period, metrics)
                    self.time_stats['prediction_adjustment'] += time.time() - adjust_start
                    self.time_counts['prediction_adjustment'] += 1

                metric_comp_start = time.time()
                mse = F.mse_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, gt, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.time_stats['metric_computation'] += time.time() - metric_comp_start
                self.time_counts['metric_computation'] += 1

                self.mse_all.append(mse)
                self.mae_all.append(mae)

                batch_start = batch_end
                batch_idx += 1

        self.time_stats['total_time'] = time.time() - total_start_time

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)

        self._save_results_to_json()
        self.model.eval()
        self.plot_lr_history()
        return self.mse_all.mean(), self.mae_all.mean()

    @torch.no_grad()
    def _adjust_prediction(self, pred, window, batch_size, period, metrics):
        window_cal = tuple(t.clone() for t in window)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            window_cal = self.cali.input_calibration(window_cal, metrics)

        pred_after, gt = forecast(self.cfg, window_cal, self.model, self.norm_module)

        if self.cfg.TTA.TAFAS.CALI_MODULE:
            pred_after = self.cali.output_calibration(pred_after, metrics)

        for i in range(batch_size - 1):
            pred[i, period - i:] = pred_after[i, period - i:]

        return pred, gt

    @torch.enable_grad()
    def _adapt_full(self):
        while self.pred_end and self.cur_step >= self.pred_end[min(self.pred_end)]:
            idx = min(self.pred_end)
            window = self.inputs_hist.pop(idx)
            self.pred_end.pop(idx)

            device = next(self.model.parameters()).device
            if self.steps_since_last_buffer_update >= self.buffer_interval:
                z, dr, dp = self._compute_and_update_buffers(window)
                self.steps_since_last_buffer_update = 0
                if self.steps_since_last_metric_update >= self.metric_interval:
                    self._update_adaptation_rate(z, dr, dp)
                    self.steps_since_last_metric_update = 0
                metrics = torch.tensor([z, dr, dp], device=device)
            else:
                metrics = torch.zeros(3, dtype=torch.float32, device=device)

            for _ in range(self.cfg.TTA.TAFAS.STEPS):
                self.n_adapt += 1
                self.switch_train()

                window_cal = tuple(t.clone() for t in window)
                if hasattr(self, 'cali'):
                    window_cal = self.cali.input_calibration(window_cal, metrics)

                pred, gt = forecast(self.cfg, window_cal, self.model, self.norm_module)

                if hasattr(self, 'cali'):
                    pred = self.cali.output_calibration(pred, metrics)

                loss = F.mse_loss(pred, gt)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.switch_eval()

    @torch.enable_grad()
    def _adapt_partial(self, window, period, batch_size, batch_idx):
        with torch.autograd.set_detect_anomaly(True):
            if self.steps_since_last_buffer_update >= self.buffer_interval:
                self._update_rtab_partial(window, period, batch_idx)

            device = next(self.model.parameters()).device
            if self.steps_since_last_metric_update >= self.metric_interval:
                z, dr, dp = self._collect_current_metrics(window)
                self._update_adaptation_rate(z, dr, dp)
                metrics = torch.tensor([z, dr, dp], device=device)
                self.steps_since_last_metric_update = 0
            else:
                metrics = torch.zeros(3, dtype=torch.float32, device=device)

            for step in range(self.cfg.TTA.TAFAS.STEPS):
                self.n_adapt += 1

                window_cal = tuple(t.clone().detach() for t in window)
                if hasattr(self, 'cali'):
                    window_cal = self.cali.input_calibration(window_cal, metrics.detach())

                pred, gt = forecast(self.cfg, window_cal, self.model, self.norm_module)

                if hasattr(self, 'cali'):
                    pred = self.cali.output_calibration(pred, metrics.detach())

                pred_p, gt_p = pred[0][:period].clone(), gt[0][:period].clone()
                loss = F.mse_loss(pred_p, gt_p)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return pred.detach(), gt.detach()

    def _collect_current_metrics(self, window):
        if len(self.mse_buffer) == 0:
            z = 0.0
        else:
            last_mse = self.mse_buffer[-1]
            mu    = np.mean(self.mse_buffer)
            sigma = np.std(self.mse_buffer)
            z     = (last_mse - mu) / (sigma + self.eps)

        dr = self._dist_rtab(window)
        dp = self._dist_rdb(window)

        return z, dr, dp

    def _compute_and_update_buffers(self, window):
        pred, gt = forecast(self.cfg, tuple(t.clone() for t in window), self.model, self.norm_module)

        mse_per_sample = F.mse_loss(pred, gt, reduction='none')          \
            .mean(dim=(-2, -1))
        mse_per_sample = mse_per_sample.detach().cpu().numpy()

        z_list = []
        for mse in mse_per_sample:
            self.mse_buffer.append(mse)
            mu = np.mean(self.mse_buffer)
            sigma = np.std(self.mse_buffer)
            z_list.append((mse - mu) / (sigma + self.eps))
        z = float(np.mean(z_list))

        batch_size = len(mse_per_sample)
        for i, mse in enumerate(mse_per_sample):
            sid = self.cur_step - batch_size + 1 + i
            emb = self._extract_embedding((
                window[0][i:i+1], window[1][i:i+1],
                window[2][i:i+1], window[3][i:i+1]
            )).detach().cpu()
            self._update_rtab_full(sid, emb, float(mse))

        dr = self._dist_rtab(window)
        dp = self._dist_rdb(window)

        return z, dr, dp

    def _update_rtab_full(self, sid, emb, mse_full):
        self.rtab[sid] = [emb, mse_full, 1.0]
        if len(self.rtab) > self.cfg.TTA.DYNATTA.RTAB_SIZE:
            oldest = min(self.rtab.keys())
            del self.rtab[oldest]
        self._update_rdb(sid, emb, mse_full)

    def _update_rtab_partial(self, window, l, idx):
        for b in range(window[0].shape[0]):
            sid = self.cur_step - window[0].shape[0] + b
            emb = self._extract_embedding((
                window[0][b:b+1], window[1][b:b+1], window[2][b:b+1], window[3][b:b+1]
            )).detach().cpu()
            single_win = (
                window[0][b:b+1], window[1][b:b+1],
                window[2][b:b+1], window[3][b:b+1]
            )
            pred_b, gt_b = forecast(self.cfg, tuple(t.clone() for t in single_win), self.model, self.norm_module)
            pred_p = pred_b[0, :l]
            gt_p   = gt_b[0, :l]
            mse_p = F.mse_loss(pred_p, gt_p).item()
            alpha = l / self.cfg.DATA.PRED_LEN
            self.rtab[sid] = [emb, mse_p, alpha]
            if len(self.rtab) > self.cfg.TTA.DYNATTA.RTAB_SIZE:
                oldest = min(self.rtab)
                del self.rtab[oldest]

    def _update_rdb(self, sid, emb, mse):
        cap = self.cfg.TTA.DYNATTA.RDB_SIZE
        if sid in self.rdb:
            if mse < self.rdb[sid][1]:
                self.rdb[sid] = [emb, mse]
        else:
            if len(self.rdb) < cap:
                self.rdb[sid] = [emb, mse]
            else:
                worst = max(self.rdb.items(), key=lambda x: x[1][1])[0]
                if mse < self.rdb[worst][1]:
                    del self.rdb[worst]
                    self.rdb[sid] = [emb, mse]

    def _dist_rtab(self, window):
        if not self.rtab: return 0.0
        embs, mses, alps = zip(*self.rtab.values())
        inv = np.array([alp / (m + self.eps) for m, alp in zip(mses, alps)], dtype=float)
        w = inv / inv.sum()
        device = next(self.model.parameters()).device
        stack = torch.stack(embs, 0).to(device)
        w_tensor = torch.from_numpy(w).to(device).view(-1, 1, 1, 1)
        avg = (stack * w_tensor).sum(0).detach().clone()
        cur = self._extract_embedding(window).detach().to(device)
        return torch.norm(cur - avg, p=2, dim=-1).mean().item()

    def _dist_rdb(self, window):
        if not self.rdb: return 0.0
        embs, mses = zip(*self.rdb.values())
        inv = np.array([1.0 / (m + self.eps) for m in mses], dtype=float)
        w = inv / inv.sum()
        device = next(self.model.parameters()).device
        stack = torch.stack(embs, 0).to(device)
        avg = (stack * torch.from_numpy(w).to(device).view(-1, 1, 1, 1)).sum(0).detach().clone()
        cur = self._extract_embedding(window).detach().to(device)
        return torch.norm(cur - avg, p=2, dim=-1).mean().item()

    def _update_adaptation_rate(self, z, dr, dp):
        norms = []
        for i, m in enumerate([z, dr, dp]):
            hist = self.metric_hist[i]
            hist.append(m)
            mu, sd = np.mean(hist), np.std(hist)
            norms.append((m - mu) / (sd + self.eps))
        S = sum(norms)
        lam = 1 + (self.alpha_max / self.alpha_min - 1) / (1 + math.exp(-self.kappa * S))
        gamma = min(1.0, self.n_adapt / (self.warmup_steps + self.eps))
        alpha_tgt = self.alpha_min * (1 + gamma * (lam - 1))
        self.alpha_t += self.eta * (alpha_tgt - self.alpha_t)
        for g in self.optimizer.param_groups:
            g['lr'] = float(self.alpha_t)
        self.lr_history.append(float(self.alpha_t))

    def _save_results_to_json(self):
        total_params = sum(p.numel() for p in self.named_params_to_adapt.values())

        avg_times = {}
        for key in self.time_stats:
            if key != 'total_time':
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count

        time_statistics = {
            "dynatta_operations": {
                "metrics_collection_ms": round(avg_times.get('metrics_collection', 0) * 1000, 3),
                "buffer_update_ms": round(avg_times.get('buffer_update', 0) * 1000, 3),
                "adaptation_rate_update_ms": round(avg_times.get('adaptation_rate_update', 0) * 1000, 3)
            },
            "adaptation_training": {
                "full_adaptation_ms": round(avg_times.get('full_adaptation', 0) * 1000, 3),
                "partial_adaptation_ms": round(avg_times.get('partial_adaptation', 0) * 1000, 3),
                "prediction_adjustment_ms": round(avg_times.get('prediction_adjustment', 0) * 1000, 3)
            },
            "other_operations": {
                "metric_computation_ms": round(avg_times.get('metric_computation', 0) * 1000, 3)
            },
            "overall_stats": {
                "total_time_seconds": round(self.time_stats['total_time'], 2),
                "total_adaptations": int(self.n_adapt),
                "avg_time_per_adaptation_ms": round(self.time_stats.get('full_adaptation', 0) / max(self.time_counts.get('full_adaptation', 1), 1) * 1000, 3),
                "throughput_samples_per_sec": round(len(self.test_loader.dataset) / self.time_stats['total_time'], 1)
            }
        }

        dynatta_metrics = {
            "mse_buffer_size": len(self.mse_buffer),
            "rtab_size": len(self.rtab),
            "rdb_size": len(self.rdb),
            "final_adaptation_rate": float(self.alpha_t),
            "alpha_min": float(self.alpha_min),
            "alpha_max": float(self.alpha_max),
            "warmup_steps": int(self.warmup_steps),
            "adaptation_rate_history_length": len(self.lr_history)
        }

        combined_results = {
            "model": "DynaTTA",
            "time_statistics": time_statistics,
            "dynatta_metrics": dynatta_metrics,
            "final_results": {
                "adaptation_count": int(self.n_adapt),
                "test_mse": float(self.mse_all.mean())
            },
            "parameters": {
                "total_params": int(total_params)
            }
        }

        print(json.dumps(combined_results, indent=2))

    def _copy_state(self):
        return deepcopy(self.model.state_dict()), deepcopy(self.optimizer.state_dict())

    def switch_train(self):
        self.model.train()
        if hasattr(self, 'cali'): self.cali.train()

    def switch_eval(self):
        self.model.eval()
        if hasattr(self, 'cali'): self.cali.eval()

    def _unfreeze_modules_to_adapt(self):
        for _, module in self.named_modules_to_adapt:
            module.requires_grad_(True)

    def _calc_period(self, enc0):
        fft = torch.fft.rfft(enc0 - enc0.mean(0), dim=0)
        amp = fft.abs(); pw = amp.pow(2).mean(0)
        try:
            per = enc0.shape[0] // fft[:, pw.argmax()].argmax().item()
        except:
            per = 24
        per *= self.cfg.TTA.TAFAS.PERIOD_N
        return per, per + 1

    def _extract_embedding(self, window):
        with torch.no_grad():
            pred, gt = forecast(self.cfg, tuple(t.clone() for t in window), self.model, self.norm_module)
        return pred.detach()

    def plot_lr_history(self):
        os.makedirs("plots", exist_ok=True)

        fig, ax = plt.subplots()
        ax.plot(self.lr_history, alpha=0.9, color="#9238B4")

        ax.set_xlabel(r'Step', fontsize=14, labelpad=10)
        ax.set_ylabel(r'Adaptation Rate', fontsize=14, labelpad=10)
        ax.set_title(f'Adaptation rate vs Step', fontsize=14, pad=10)

        ax.set_facecolor('#EAEAF2')
        ax.legend(facecolor="#EAEAF2", prop={'size': 8}, loc='upper right')

        ax.tick_params(axis=u'both', which=u'both',length=0, pad=9.5, labelsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.ticklabel_format(useOffset=False)

        ax.grid(True, color='white')
        fig.tight_layout()
        plt.savefig(f"plots/plot_lr_history_DYNATTA_{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_{self.cfg.DATA.PRED_LEN}_warmup{self.cfg.TTA.DYNATTA.WARMUP_FACTOR}_buffer_update_step_{self.cfg.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL}.pdf",bbox_inches='tight', pad_inches=0.0)
        plt.savefig(f"plots/plot_lr_history_DYNATTA_{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_{self.cfg.DATA.PRED_LEN}_warmup{self.cfg.TTA.DYNATTA.WARMUP_FACTOR}_buffer_update_step_{self.cfg.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL}.png",bbox_inches='tight', pad_inches=0.0)

        import pandas as pd
        pd.DataFrame(self.lr_history, columns=["lr"]).to_csv(
            f"plots/lr_history_{self.cfg.MODEL.NAME}_{self.cfg.DATA.NAME}_{self.cfg.DATA.PRED_LEN}_warmup{self.cfg.TTA.DYNATTA.WARMUP_FACTOR}_buffer_update_step_{self.cfg.TTA.DYNATTA.UPDATE_BUFFERS_INTERVAL}.csv",
            index_label="step"
        )


class DynamicGCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden=64, gating_init=0.01, var_wise=True, metric_dim=3):
        super().__init__()
        self.var_wise = var_wise
        if var_wise:
            self.weight = nn.Parameter(torch.zeros(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.zeros(window_len, window_len))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))
        self.static_g = nn.Parameter(gating_init * torch.ones(n_var))
        self.mlp = nn.Sequential(
            nn.Linear(metric_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_var)
        )

    def forward(self, x, metrics):
        if metrics.dim() == 1:
            m = metrics.unsqueeze(0).float()
        else:
            m = metrics.clone()
        adj = self.mlp(m).mean(0)
        g = torch.tanh(self.static_g + adj).view(1,1,-1)
        if self.var_wise:
            cal = x + g * (torch.einsum('biv,iov->bov', x, self.weight) + self.bias)
        else:
            cal = x + g * (torch.einsum('biv,io->bov', x, self.weight) + self.bias)
        return cal


class Calibration(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        seq, pred, n_var = cfg.DATA.SEQ_LEN, cfg.DATA.PRED_LEN, cfg.DATA.N_VAR
        hd, init, vw = cfg.TTA.TAFAS.HIDDEN_DIM, cfg.TTA.TAFAS.GATING_INIT, cfg.TTA.TAFAS.GCM_VAR_WISE
        dim = 3
        if cfg.MODEL.NAME == 'PatchTST':
            self.in_cali = DynamicGCM(seq, 1, hd, init, vw, dim)
            self.out_cali = DynamicGCM(pred, 1, hd, init, vw, dim)
        else:
            self.in_cali = DynamicGCM(seq, n_var, hd, init, vw, dim)
            self.out_cali = DynamicGCM(pred, n_var, hd, init, vw, dim)

    def input_calibration(self, window, metrics=None):
        x_enc, x_mark, x_dec, d_mark = prepare_inputs(window)
        return (self.in_cali(x_enc.clone(), metrics), x_mark, x_dec, d_mark)

    def output_calibration(self, out, metrics=None):
        return self.out_cali(out.clone(), metrics)


def build_adapter(cfg, model, norm_module=None):
    return DynaTTAAdapter(cfg, model, norm_module)