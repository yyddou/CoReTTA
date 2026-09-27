# adapted from tafas.py - adds SCR cross-variate correction module

from typing import List
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.optimizer import get_optimizer
from models.forecast import forecast
from datasets.loader import get_test_dataloader
from utils.misc import prepare_inputs
from config import get_norm_method


class Adapter(nn.Module):
    def __init__(self, cfg, model: nn.Module, norm_module=None):
        super(Adapter, self).__init__()
        self.cfg = cfg
        self.model_cfg = cfg.MODEL
        self.model = model
        self.norm_method = get_norm_method(cfg)
        self.norm_module = norm_module
        self.test_loader = get_test_dataloader(cfg)
        self.test_data = self.test_loader.dataset.test

        if self.cfg.TTA.TAFAS.CALI_MODULE:
            self.cali = Calibration(cfg).cuda()

        self._freeze_all_model_params()
        self.named_modules_to_adapt = self._get_named_modules_to_adapt()
        self._unfreeze_modules_to_adapt()
        self.named_params_to_adapt = self._get_named_params_to_adapt()

        self.optimizer = get_optimizer(self.named_params_to_adapt.values(), cfg.TTA)

        # -- SCR separate optimizer ----------------------------------------
        # STAR optimizer with 10x learning rate
        import copy
        star_cfg = copy.deepcopy(cfg.TTA)
        star_cfg.SOLVER.BASE_LR = cfg.TTA.SOLVER.BASE_LR * 10.0
        self.star_optimizer   = get_optimizer(self.cali.star_parameters(), star_cfg)
        self.star_adapt_steps = getattr(cfg.TTA.COSA, "STEPS", 20)

        self.model_state, self.optimizer_state, self.star_optimizer_state = self._copy_model_and_optimizer()

        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []

        # -- Spectral context ----------------------------------------------
        self._cur_spec_entropy = 0.0
        self._cur_low_band     = 0.0
        self._cur_mid_band     = 0.0
        self._cur_high_band    = 0.0
        self._cur_centroid     = 0.0
        self._ref_low_band     = None
        self._ref_mid_band     = None
        self._ref_centroid     = None
        self._ref_alpha        = 0.05

    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError

    def reset(self):
        self._load_model_and_optimizer()

    def _copy_model_and_optimizer(self):
        model_state     = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        star_opt_state  = deepcopy(self.star_optimizer.state_dict())
        return model_state, optimizer_state, star_opt_state

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.star_optimizer.load_state_dict(self.star_optimizer_state)

    def _get_all_models(self):
        models = [self.model]
        if self.norm_module is not None:
            models.append(self.norm_module)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            models.append(self.cali)
        return models

    def _freeze_all_model_params(self):
        for model in self._get_all_models():
            for param in model.parameters():
                param.requires_grad_(False)

    def _get_named_modules(self):
        named_modules = []
        for model in self._get_all_models():
            named_modules += list(model.named_modules())
        return named_modules

    def _get_named_modules_to_adapt(self) -> List[str]:
        named_modules = self._get_named_modules()
        if self.cfg.TTA.MODULE_NAMES_TO_ADAPT == "all":
            return named_modules
        named_modules_to_adapt = []
        for module_name in self.cfg.TTA.MODULE_NAMES_TO_ADAPT.split(","):
            exact_match = "(exact)" in module_name
            module_name = module_name.replace("(exact)", "")
            if exact_match:
                named_modules_to_adapt += [(n, m) for n, m in named_modules if n == module_name]
            else:
                named_modules_to_adapt += [(n, m) for n, m in named_modules if module_name in n]
        assert len(named_modules_to_adapt) > 0
        return named_modules_to_adapt

    def _unfreeze_modules_to_adapt(self):
        for _, module in self.named_modules_to_adapt:
            module.requires_grad_(True)

    def _get_named_params_to_adapt(self):
        named_params_to_adapt = {}
        for model in self._get_all_models():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    named_params_to_adapt[name] = param
        return named_params_to_adapt

    def switch_model_to_train(self):
        for model in self._get_all_models():
            model.train()

    def switch_model_to_eval(self):
        for model in self._get_all_models():
            model.eval()

    def _compute_spectral_context(self, enc_window: torch.Tensor):
        eps   = 1e-8
        x     = enc_window - enc_window.mean(dim=1, keepdim=True)
        fft   = torch.fft.rfft(x, dim=1)
        pow_  = (fft.real**2 + fft.imag**2).mean(dim=2).mean(dim=0)
        K     = pow_.shape[0]
        pi    = (pow_ + eps) / (pow_.sum() + eps * K)
        H     = float(-(pi * torch.log(pi + eps)).sum() / np.log(max(K, 2)))
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

    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude  = torch.abs(fft_result)
        power      = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.TAFAS.PERIOD_N
        batch_size = period + 1
        return period, batch_size

    @torch.enable_grad()
    def adapt_tafas(self):
        batch_start = 0
        batch_end   = 0
        batch_idx   = 0
        is_last     = False

        self.switch_model_to_eval()
        for idx, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)
            while batch_end < len(enc_window_all):
                enc_window_first = enc_window_all[batch_start]

                if self.cfg.TTA.TAFAS.PAAS:
                    period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
                else:
                    batch_size = self.cfg.TTA.TAFAS.BATCH_SIZE
                    period     = batch_size - 1

                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end  = len(enc_window_all)
                    batch_size = batch_end - batch_start
                    is_last    = True

                self.cur_step += batch_size

                inputs = (
                    enc_window_all[batch_start:batch_end],
                    enc_window_stamp_all[batch_start:batch_end],
                    dec_window_all[batch_start:batch_end],
                    dec_window_stamp_all[batch_start:batch_end],
                )

                # -- spectral context --------------------------------------
                self._compute_spectral_context(enc_window_all[batch_start:batch_end])
                spec_ctx = torch.tensor(
                    [[self._cur_spec_entropy, self._cur_low_band,
                      self._cur_mid_band, self._cur_high_band]],
                    dtype=torch.float32, device="cuda"
                ).expand(batch_size, -1)

                self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_dict[batch_idx]        = inputs

                self._adapt_with_full_ground_truth_if_available(spec_ctx=spec_ctx)
                pred, ground_truth = self._adapt_with_partial_ground_truth(
                    inputs, period, batch_size, batch_idx, spec_ctx=spec_ctx)

                if self.cfg.TTA.TAFAS.ADJUST_PRED:
                    pred, ground_truth = self._adjust_prediction(
                        pred, inputs, batch_size, period, spec_ctx=spec_ctx)

                mse = F.mse_loss(pred, ground_truth, reduction="none").mean(dim=(-2,-1)).detach().cpu().numpy()
                mae = F.l1_loss( pred, ground_truth, reduction="none").mean(dim=(-2,-1)).detach().cpu().numpy()
                self.mse_all.append(mse)
                self.mae_all.append(mae)

                batch_start = batch_end
                batch_idx  += 1

        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1
        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)

        print("After TSF-TTA of TAFAS+CoRe")
        print(f"Number of adaptations: {self.n_adapt}")
        print(f"Test MSE: {self.mse_all.mean():.4f}, Test MAE: {self.mae_all.mean():.4f}")

        self.model.eval()

    def _adapt_with_full_ground_truth_if_available(self, spec_ctx=None):
        while self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)

            # -- TAFAS original: GCM step=1 ---------------------------------
            for _ in range(self.cfg.TTA.TAFAS.STEPS):
                self.n_adapt += 1
                self.switch_model_to_train()
                if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != "PatchTST":
                    inputs_history = self.cali.input_calibration(inputs_history)
                pred, ground_truth = forecast(self.cfg, inputs_history, self.model, self.norm_module)
                if self.cfg.TTA.TAFAS.CALI_MODULE:
                    pred = self.cali.output_calibration(pred, spec_ctx=spec_ctx)
                loss = F.mse_loss(pred, ground_truth)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.switch_model_to_eval()

            # -- SCR extra steps -------------------------------------------
            with torch.no_grad():
                pred_s, gt_s = forecast(self.cfg, inputs_history, self.model, self.norm_module)
                pred_s = pred_s.detach()
                gt_s   = gt_s.detach()
            for _ in range(self.star_adapt_steps):
                self.switch_model_to_train()
                cal_s  = self.cali.output_calibration(pred_s, spec_ctx=spec_ctx)
                loss_s = F.mse_loss(cal_s, gt_s)
                self.star_optimizer.zero_grad()
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(self.cali.star_parameters(), max_norm=0.1)
                self.star_optimizer.step()
                self.switch_model_to_eval()

            self.pred_step_end_dict.pop(batch_idx_available)

    def _adapt_with_partial_ground_truth(self, inputs, period, batch_size, batch_idx, spec_ctx=None):
        for _ in range(self.cfg.TTA.TAFAS.STEPS):
            self.n_adapt += 1
            if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != "PatchTST":
                inputs = self.cali.input_calibration(inputs)
            pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
            if self.cfg.TTA.TAFAS.CALI_MODULE:
                pred = self.cali.output_calibration(pred, spec_ctx=spec_ctx)
            pred_partial, ground_truth_partial = pred[0][:period], ground_truth[0][:period]
            mse_partial = F.mse_loss(pred_partial, ground_truth_partial)
            self.optimizer.zero_grad()
            mse_partial.backward()
            self.optimizer.step()

        # -- SCR extra steps -----------------------------------------------
        if pred is not None and ground_truth is not None:
            _pred_detach = pred.detach()
            _gt_detach   = ground_truth.detach()
            for _ in range(self.star_adapt_steps):
                self.switch_model_to_train()
                cal_s  = self.cali.output_calibration(_pred_detach, spec_ctx=spec_ctx)
                loss_s = F.mse_loss(cal_s, _gt_detach)
                self.star_optimizer.zero_grad()
                loss_s.backward()
                torch.nn.utils.clip_grad_norm_(self.cali.star_parameters(), max_norm=0.1)
                self.star_optimizer.step()
                self.switch_model_to_eval()

        return pred, ground_truth

    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period, spec_ctx=None):
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            inputs = self.cali.input_calibration(inputs)
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            B = pred_after_adapt.shape[0]
            _ctx = spec_ctx[:1].expand(B, -1) if spec_ctx is not None else None
            pred_after_adapt = self.cali.output_calibration(pred_after_adapt, spec_ctx=_ctx)
        for i in range(batch_size-1):
            pred[i, period-i:] = pred_after_adapt[i, period-i:]
        return pred, ground_truth

    def adapt(self):
        self.adapt_tafas()


def build_adapter(cfg, model, norm_module=None):
    adapter = Adapter(cfg, model, norm_module)
    return adapter


class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True, use_star=False):
        super(GCM, self).__init__()
        self.window_len = window_len
        self.n_var      = n_var
        self.var_wise   = var_wise
        self.use_star   = use_star
        if var_wise:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.weight.data.zero_()
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias   = nn.Parameter(torch.zeros(window_len, n_var))

        # -- SCR on GCM correction space -----------------------------------
        if use_star:
            T, V = window_len, n_var
            _r = V
            self.star_down     = nn.Linear(2 * T, _r)
            self.star_up       = nn.Linear(_r, T)
            self.star_gate_net = nn.Linear(4, V)
            nn.init.xavier_uniform_(self.star_down.weight, gain=0.01)
            nn.init.zeros_(self.star_down.bias)
            nn.init.xavier_uniform_(self.star_up.weight, gain=0.01)
            nn.init.zeros_(self.star_up.bias)
            nn.init.zeros_(self.star_gate_net.weight)
            nn.init.constant_(self.star_gate_net.bias, -1.0)

    def star_parameters(self):
        if not self.use_star:
            return []
        return [self.star_down.weight, self.star_down.bias,
                self.star_up.weight,   self.star_up.bias,
                self.star_gate_net.weight, self.star_gate_net.bias]

    def forward(self, x, spec_ctx=None):
        if self.var_wise:
            correction = torch.einsum("biv,iov->bov", x, self.weight) + self.bias
        else:
            correction = torch.einsum("biv,io->bov", x, self.weight) + self.bias

        # -- SCR on correction space ---------------------------------------
        if self.use_star:
            B = x.shape[0]
            corr_T  = correction.permute(0, 2, 1)          # [B, V, T]
            core    = corr_T.mean(dim=1, keepdim=True)      # [B, 1, T]
            core_e  = core.expand_as(corr_T)                # [B, V, T]
            delta   = self.star_up(
                torch.tanh(self.star_down(torch.cat([corr_T, core_e], dim=-1)))
            ).permute(0, 2, 1)                              # [B, T, V]
            freq_ctx_g = spec_ctx[:1].expand(B, -1) if spec_ctx is not None else torch.zeros(B, 4, device=x.device)
            gate    = torch.tanh(self.star_gate_net(freq_ctx_g)).unsqueeze(1)  # [B, 1, V]
            correction = correction + gate * delta

        return x + torch.tanh(self.gating) * correction


class Calibration(nn.Module):
    def __init__(self, cfg):
        super(Calibration, self).__init__()
        self.cfg       = cfg
        self.seq_len   = cfg.DATA.SEQ_LEN
        self.pred_len  = cfg.DATA.PRED_LEN
        self.n_var     = cfg.DATA.N_VAR
        self.hidden_dim  = cfg.TTA.TAFAS.HIDDEN_DIM
        self.gating_init = cfg.TTA.TAFAS.GATING_INIT
        self.var_wise    = cfg.TTA.TAFAS.GCM_VAR_WISE

        if cfg.MODEL.NAME == "PatchTST":
            self.in_cali  = GCM(self.seq_len,  1, self.hidden_dim, self.gating_init, self.var_wise, use_star=False)
            self.out_cali = GCM(self.pred_len,  1, self.hidden_dim, self.gating_init, self.var_wise, use_star=True)
        else:
            self.in_cali  = GCM(self.seq_len,  self.n_var, self.hidden_dim, self.gating_init, self.var_wise, use_star=False)
            self.out_cali = GCM(self.pred_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise, use_star=True)

    def star_parameters(self):
        return self.out_cali.star_parameters()

    def input_calibration(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        enc_window = self.in_cali(enc_window)
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp

    def output_calibration(self, outputs, spec_ctx=None):
        # TAFAS GCM with STAR inside correction space
        return self.out_cali(outputs, spec_ctx=spec_ctx)
