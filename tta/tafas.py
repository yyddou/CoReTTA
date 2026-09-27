##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

# adapted from https://github.com/DequanWang/tent/blob/master/tent.py

from typing import List
from copy import deepcopy
from collections import defaultdict
import time
import json
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
            if torch.cuda.is_available():
                self.cali = Calibration(cfg).cuda()
            else:
                self.cali = Calibration(cfg)

        self._freeze_all_model_params()
        self.named_modules_to_adapt = self._get_named_modules_to_adapt()
        self._unfreeze_modules_to_adapt()
        self.named_params_to_adapt = self._get_named_params_to_adapt()

        self.optimizer = get_optimizer(self.named_params_to_adapt.values(), cfg.TTA)

        self.model_state, self.optimizer_state = self._copy_model_and_optimizer()

        cfg.TEST.BATCH_SIZE = len(self.test_loader.dataset)
        self.test_loader = get_test_dataloader(cfg)
        self.cur_step = cfg.DATA.SEQ_LEN - 2
        self.pred_step_end_dict = {}
        self.inputs_dict = {}
        self.n_adapt = 0

        self.mse_all = []
        self.mae_all = []
        self.time_stats = defaultdict(float)
        self.time_counts = defaultdict(int)

    def forward(self, enc_window, enc_window_stamp, dec_window, dec_window_stamp):
        raise NotImplementedError

    def reset(self):
        self._load_model_and_optimizer()

    def count_parameters(self):
        trainable_params = []
        total_sum = 0

        for name, param in self.cali.named_parameters():
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
            "model": "TAFAS",
            "parameters": {
                "trainable_params": trainable_params,
                "total_params": total_sum
            }
        }


    def _copy_model_and_optimizer(self):
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        return model_state, optimizer_state

    def _load_model_and_optimizer(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)

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

    @torch.enable_grad()
    def adapt_tafas(self):
        batch_start = 0
        batch_end = 0
        batch_idx = 0
        is_last = False
        test_len = len(self.test_loader.dataset)

        total_start_time = time.time()

        self.switch_model_to_eval()
        for idx, inputs in enumerate(self.test_loader):
            enc_window_all, enc_window_stamp_all, dec_window_all, dec_window_stamp_all = prepare_inputs(inputs)

            while batch_end < len(enc_window_all):
                batch_process_start = time.time()

                enc_window_first = enc_window_all[batch_start]

                if self.cfg.TTA.TAFAS.PAAS:
                    period_start = time.time()
                    period, batch_size = self._calculate_period_and_batch_size(enc_window_first)
                    self.time_stats['period_calculation'] += time.time() - period_start
                    self.time_counts['period_calculation'] += 1
                else:
                    batch_size = self.cfg.TTA.TAFAS.BATCH_SIZE
                    period = batch_size - 1

                batch_end = batch_start + batch_size
                if batch_end > len(enc_window_all):
                    batch_end = len(enc_window_all)
                    batch_size = batch_end - batch_start
                    is_last = True

                self.cur_step += batch_size

                inputs = enc_window_all[batch_start:batch_end], enc_window_stamp_all[batch_start:batch_end], \
                         dec_window_all[batch_start:batch_end], dec_window_stamp_all[batch_start:batch_end]

                self.pred_step_end_dict[batch_idx] = self.cur_step + self.cfg.DATA.PRED_LEN
                self.inputs_dict[batch_idx] = inputs

                full_adapt_start = time.time()
                self._adapt_with_full_ground_truth_if_available()
                self.time_stats['full_adaptation_total'] += time.time() - full_adapt_start
                if self.time_stats['full_adaptation_count'] > 0:
                    self.time_counts['full_adaptation_total'] += 1

                partial_adapt_start = time.time()
                pred, ground_truth = self._adapt_with_partial_ground_truth(inputs, period, batch_size, batch_idx)
                self.time_stats['partial_adaptation_total'] += time.time() - partial_adapt_start
                self.time_counts['partial_adaptation_total'] += 1

                if self.cfg.TTA.TAFAS.ADJUST_PRED:
                    adjust_start = time.time()
                    pred, ground_truth = self._adjust_prediction(pred, inputs, batch_size, period)
                    self.time_stats['prediction_adjustment'] += time.time() - adjust_start
                    self.time_counts['prediction_adjustment'] += 1

                metric_start = time.time()
                mse = F.mse_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                mae = F.l1_loss(pred, ground_truth, reduction='none').mean(dim=(-2, -1)).detach().cpu().numpy()
                self.time_stats['metric_computation'] += time.time() - metric_start
                self.time_counts['metric_computation'] += 1

                self.mse_all.append(mse)
                self.mae_all.append(mae)

                self.time_stats['batch_processing'] += time.time() - batch_process_start
                self.time_counts['batch_processing'] += 1

                batch_start = batch_end
                batch_idx += 1

        self.time_stats['total_time'] = time.time() - total_start_time

        assert self.cur_step == len(self.test_data) - self.cfg.DATA.PRED_LEN - 1

        self.mse_all = np.concatenate(self.mse_all)
        self.mae_all = np.concatenate(self.mae_all)
        assert len(self.mse_all) == len(self.test_loader.dataset)

        self._print_combined_results()

        self.model.eval()

    def _adapt_with_full_ground_truth_if_available(self):
        adapted_count = 0

        while self.cur_step >= self.pred_step_end_dict[min(self.pred_step_end_dict.keys())]:
            batch_idx_available = min(self.pred_step_end_dict.keys())
            inputs_history = self.inputs_dict.pop(batch_idx_available)

            for _ in range(self.cfg.TTA.TAFAS.STEPS):
                step_start = time.time()
                self.n_adapt += 1
                adapted_count += 1

                self.switch_model_to_train()

                if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
                    cali_start = time.time()
                    inputs_history = self.cali.input_calibration(inputs_history)
                    self.time_stats['full_calibration'] += time.time() - cali_start
                    self.time_counts['full_calibration'] += 1

                forward_start = time.time()
                pred, ground_truth = forecast(self.cfg, inputs_history, self.model, self.norm_module)

                if self.cfg.TTA.TAFAS.CALI_MODULE:
                    cali_out_start = time.time()
                    pred = self.cali.output_calibration(pred)
                    self.time_stats['full_calibration'] += time.time() - cali_out_start
                    self.time_counts['full_calibration'] += 1

                self.time_stats['full_forward_pass'] += time.time() - forward_start
                self.time_counts['full_forward_pass'] += 1

                loss_start = time.time()
                loss = F.mse_loss(pred, ground_truth)
                self.time_stats['full_loss_computation'] += time.time() - loss_start
                self.time_counts['full_loss_computation'] += 1

                backward_start = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.time_stats['full_backward_update'] += time.time() - backward_start
                self.time_counts['full_backward_update'] += 1

                self.switch_model_to_eval()

                self.time_stats['full_adaptation_step'] += time.time() - step_start
                self.time_counts['full_adaptation_step'] += 1

            self.pred_step_end_dict.pop(batch_idx_available)

        if adapted_count > 0:
            self.time_stats['full_adaptation_count'] = adapted_count

    def _adapt_with_partial_ground_truth(self, inputs, period, batch_size, batch_idx):
        pred, ground_truth = None, None

        for _ in range(self.cfg.TTA.TAFAS.STEPS):
            step_start = time.time()
            self.n_adapt += 1

            if self.cfg.TTA.TAFAS.CALI_MODULE and self.cfg.MODEL.NAME != 'PatchTST':
                cali_start = time.time()
                inputs = self.cali.input_calibration(inputs)
                self.time_stats['partial_calibration'] += time.time() - cali_start
                self.time_counts['partial_calibration'] += 1

            forward_start = time.time()
            pred, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)

            if self.cfg.TTA.TAFAS.CALI_MODULE:
                cali_out_start = time.time()
                pred = self.cali.output_calibration(pred)
                self.time_stats['partial_calibration'] += time.time() - cali_out_start
                self.time_counts['partial_calibration'] += 1

            self.time_stats['partial_forward_pass'] += time.time() - forward_start
            self.time_counts['partial_forward_pass'] += 1

            pred_partial, ground_truth_partial = pred[0][:period], ground_truth[0][:period]

            loss_start = time.time()
            mse_partial = F.mse_loss(pred_partial, ground_truth_partial)
            self.time_stats['partial_loss_computation'] += time.time() - loss_start
            self.time_counts['partial_loss_computation'] += 1

            backward_start = time.time()
            self.optimizer.zero_grad()
            mse_partial.backward()
            self.optimizer.step()
            self.time_stats['partial_backward_update'] += time.time() - backward_start
            self.time_counts['partial_backward_update'] += 1

            self.time_stats['partial_adaptation_step'] += time.time() - step_start
            self.time_counts['partial_adaptation_step'] += 1

        return pred, ground_truth

    def _print_time_statistics(self):
        avg_times = {}
        for key in self.time_stats:
            if key != 'total_time' and 'count' not in key:
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count

        time_statistics = {
            "model": "TAFAS",
            "time_statistics": {
                "full_adaptation": {},
                "partial_adaptation": {},
                "other_operations": {},
                "overall_stats": {}
            }
        }

        if 'full_adaptation_step' in avg_times:
            time_statistics["time_statistics"]["full_adaptation"] = {
                "calibration_ms": round(avg_times.get('full_calibration', 0) * 1000, 3),
                "forward_pass_ms": round(avg_times.get('full_forward_pass', 0) * 1000, 3),
                "loss_computation_ms": round(avg_times.get('full_loss_computation', 0) * 1000, 3),
                "backward_update_ms": round(avg_times.get('full_backward_update', 0) * 1000, 3),
                "per_step_total_ms": round(avg_times.get('full_adaptation_step', 0) * 1000, 3),
                "total_per_batch_ms": round(avg_times.get('full_adaptation_total', 0) * 1000, 3),
                "full_adaptation_count": int(self.time_stats.get('full_adaptation_count', 0))
            }
        else:
            time_statistics["time_statistics"]["full_adaptation"] = {
                "status": "No full adaptations performed"
            }

        time_statistics["time_statistics"]["partial_adaptation"] = {
            "calibration_ms": round(avg_times.get('partial_calibration', 0) * 1000, 3),
            "forward_pass_ms": round(avg_times.get('partial_forward_pass', 0) * 1000, 3),
            "loss_computation_ms": round(avg_times.get('partial_loss_computation', 0) * 1000, 3),
            "backward_update_ms": round(avg_times.get('partial_backward_update', 0) * 1000, 3),
            "per_step_total_ms": round(avg_times.get('partial_adaptation_step', 0) * 1000, 3),
            "total_per_batch_ms": round(avg_times.get('partial_adaptation_total', 0) * 1000, 3)
        }

        other_ops = {}
        if self.cfg.TTA.TAFAS.PAAS:
            other_ops["period_calculation_ms"] = round(avg_times.get('period_calculation', 0) * 1000, 3)
        if self.cfg.TTA.TAFAS.ADJUST_PRED:
            other_ops["prediction_adjustment_ms"] = round(avg_times.get('prediction_adjustment', 0) * 1000, 3)
        other_ops["metric_computation_ms"] = round(avg_times.get('metric_computation', 0) * 1000, 3)
        other_ops["total_batch_processing_ms"] = round(avg_times.get('batch_processing', 0) * 1000, 3)
        time_statistics["time_statistics"]["other_operations"] = other_ops

        time_statistics["time_statistics"]["overall_stats"] = {
            "total_time_seconds": round(self.time_stats['total_time'], 2),
            "total_adaptations": int(self.n_adapt),
            "number_of_batches": int(self.time_counts.get('batch_processing', 0)),
            "avg_time_per_adaptation_ms": round((self.time_stats.get('full_adaptation_step', 0) + self.time_stats.get('partial_adaptation_step', 0))/max(self.n_adapt, 1)*1000, 3),
            "throughput_samples_per_sec": round(len(self.test_loader.dataset)/self.time_stats['total_time'], 1)
        }


    def _print_combined_results(self):
        total_params = 0
        for name, param in self.cali.named_parameters():
            if param.requires_grad:
                total_params += int(param.numel())

        avg_times = {}
        for key in self.time_stats:
            if key != 'total_time' and 'count' not in key:
                count = self.time_counts[key] if self.time_counts[key] > 0 else 1
                avg_times[key] = self.time_stats[key] / count

        time_statistics = {
            "full_adaptation": {},
            "partial_adaptation": {},
            "other_operations": {},
            "overall_stats": {}
        }

        if 'full_adaptation_step' in avg_times:
            time_statistics["full_adaptation"] = {
                "calibration_ms": round(avg_times.get('full_calibration', 0) * 1000, 3),
                "forward_pass_ms": round(avg_times.get('full_forward_pass', 0) * 1000, 3),
                "loss_computation_ms": round(avg_times.get('full_loss_computation', 0) * 1000, 3),
                "backward_update_ms": round(avg_times.get('full_backward_update', 0) * 1000, 3),
                "per_step_total_ms": round(avg_times.get('full_adaptation_step', 0) * 1000, 3),
                "total_per_batch_ms": round(avg_times.get('full_adaptation_total', 0) * 1000, 3),
                "full_adaptation_count": int(self.time_stats.get('full_adaptation_count', 0))
            }
        else:
            time_statistics["full_adaptation"] = {
                "status": "No full adaptations performed"
            }

        time_statistics["partial_adaptation"] = {
            "calibration_ms": round(avg_times.get('partial_calibration', 0) * 1000, 3),
            "forward_pass_ms": round(avg_times.get('partial_forward_pass', 0) * 1000, 3),
            "loss_computation_ms": round(avg_times.get('partial_loss_computation', 0) * 1000, 3),
            "backward_update_ms": round(avg_times.get('partial_backward_update', 0) * 1000, 3),
            "per_step_total_ms": round(avg_times.get('partial_adaptation_step', 0) * 1000, 3),
            "total_per_batch_ms": round(avg_times.get('partial_adaptation_total', 0) * 1000, 3)
        }

        other_ops = {}
        if self.cfg.TTA.TAFAS.PAAS:
            other_ops["period_calculation_ms"] = round(avg_times.get('period_calculation', 0) * 1000, 3)
        if self.cfg.TTA.TAFAS.ADJUST_PRED:
            other_ops["prediction_adjustment_ms"] = round(avg_times.get('prediction_adjustment', 0) * 1000, 3)
        other_ops["metric_computation_ms"] = round(avg_times.get('metric_computation', 0) * 1000, 3)
        other_ops["total_batch_processing_ms"] = round(avg_times.get('batch_processing', 0) * 1000, 3)
        time_statistics["other_operations"] = other_ops

        time_statistics["overall_stats"] = {
            "total_time_seconds": round(self.time_stats['total_time'], 2),
            "total_adaptations": int(self.n_adapt),
            "number_of_batches": int(self.time_counts.get('batch_processing', 0)),
            "avg_time_per_adaptation_ms": round((self.time_stats.get('full_adaptation_step', 0) + self.time_stats.get('partial_adaptation_step', 0))/max(self.n_adapt, 1)*1000, 3),
            "throughput_samples_per_sec": round(len(self.test_loader.dataset)/self.time_stats['total_time'], 1)
        }

        combined_results = {
            "model": "TAFAS",
            "time_statistics": time_statistics,
            "final_results": {
                "adaptation_count": int(self.n_adapt),
                "test_mse": float(self.mse_all.mean())
            },
            "parameters": {
                "total_params": total_params
            }
        }

        print(json.dumps(combined_results, indent=2))

        from pathlib import Path
        import json as _json
        result_dir = Path(self.cfg.RESULT_DIR)
        result_dir.mkdir(parents=True, exist_ok=True)
        with open(result_dir / "tafas_result.json", "w") as f:
            _json.dump(combined_results, f, indent=2)
        with open(result_dir / "cosa_adapt_true.txt", "w") as f:
            f.write(
                f"adapt_test_mse: {float(self.mse_all.mean()):.6f}\n"
                f"adapt_test_mae: {float(self.mae_all.mean()):.6f}\n"
                f"n_adapt: {int(self.n_adapt)}\n"
            )
        import pandas as pd
        pd.DataFrame({"batch_mse": self.mse_all}).to_csv(
            f"{self.cfg.RESULT_DIR}/per_batch_mse.csv", index=False)

    def _calculate_period_and_batch_size(self, enc_window_first):
        fft_result = torch.fft.rfft(enc_window_first - enc_window_first.mean(dim=0), dim=0)
        amplitude = torch.abs(fft_result)
        power = torch.mean(amplitude ** 2, dim=0)
        try:
            period = enc_window_first.shape[0] // torch.argmax(amplitude[:, power.argmax()]).item()
        except:
            period = 24
        period *= self.cfg.TTA.TAFAS.PERIOD_N
        batch_size = period + 1
        return period, batch_size


    @torch.no_grad()
    def _adjust_prediction(self, pred, inputs, batch_size, period):
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            inputs = self.cali.input_calibration(inputs)
        pred_after_adapt, ground_truth = forecast(self.cfg, inputs, self.model, self.norm_module)
        if self.cfg.TTA.TAFAS.CALI_MODULE:
            pred_after_adapt = self.cali.output_calibration(pred_after_adapt)

        for i in range(batch_size-1):
            pred[i, period-i:] = pred_after_adapt[i, period-i:]

        return pred, ground_truth

    def adapt(self):
        self.adapt_tafas()

def build_adapter(cfg, model, norm_module=None):
    adapter = Adapter(cfg, model, norm_module)
    return adapter


class GCM(nn.Module):
    def __init__(self, window_len, n_var=1, hidden_dim=64, gating_init=0.01, var_wise=True):
        super(GCM, self).__init__()
        self.window_len = window_len
        self.n_var = n_var
        self.var_wise = var_wise
        if var_wise:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len, n_var))
        else:
            self.weight = nn.Parameter(torch.Tensor(window_len, window_len))
        self.weight.data.zero_()
        self.gating = nn.Parameter(gating_init * torch.ones(n_var))
        self.bias = nn.Parameter(torch.zeros(window_len, n_var))

    def forward(self, x):
        if self.var_wise:
            x = x + torch.tanh(self.gating) * (torch.einsum('biv,iov->bov', x, self.weight) + self.bias)
        else:
            x = x + torch.tanh(self.gating) * (torch.einsum('biv,io->bov', x, self.weight) + self.bias)
        return x


class Calibration(nn.Module):
    def __init__(self, cfg):
        super(Calibration, self).__init__()
        self.cfg = cfg
        self.seq_len = cfg.DATA.SEQ_LEN
        self.pred_len = cfg.DATA.PRED_LEN
        self.n_var = cfg.DATA.N_VAR
        self.hidden_dim = cfg.TTA.TAFAS.HIDDEN_DIM
        self.gating_init = cfg.TTA.TAFAS.GATING_INIT
        self.var_wise = cfg.TTA.TAFAS.GCM_VAR_WISE
        if cfg.MODEL.NAME == 'PatchTST':
            self.in_cali = GCM(self.seq_len, 1, self.hidden_dim, self.gating_init, self.var_wise)
            self.out_cali = GCM(self.pred_len, 1, self.hidden_dim, self.gating_init, self.var_wise)
        else:
            self.in_cali = GCM(self.seq_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise)
            self.out_cali = GCM(self.pred_len, self.n_var, self.hidden_dim, self.gating_init, self.var_wise)

    def input_calibration(self, inputs):
        enc_window, enc_window_stamp, dec_window, dec_window_stamp = prepare_inputs(inputs)
        enc_window = self.in_cali(enc_window)
        return enc_window, enc_window_stamp, dec_window, dec_window_stamp

    def output_calibration(self, outputs):
        return self.out_cali(outputs)