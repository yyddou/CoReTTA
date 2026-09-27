##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

import os
from typing import Optional

import torch

from models import iTransformer, PatchTST, DLinear, OLS, FreTS, MICN, Informer 


def build_model(cfg):
    assert cfg.MODEL.NAME in globals(), f"model {cfg.MODEL.NAME} is not defined"
    model_class = getattr(globals()[cfg.MODEL.NAME], "Model")
    model = model_class(cfg.MODEL)

    if torch.cuda.is_available():
        model = model.cuda()

    return model


def build_norm_module(cfg):
    norm_module_name = cfg.NORM_MODULE.NAME
    if norm_module_name == "RevIN":
        from models.RevIN import RevIN
        norm_module = RevIN(cfg)
    elif norm_module_name == "SAN":
        from models.Statistics_prediction import Statistics_prediction
        norm_module = Statistics_prediction(cfg)
    elif norm_module_name == "DishTS":
        from models.DishTS import DishTS
        norm_module = DishTS(cfg)
    else:
        raise ValueError

    if torch.cuda.is_available():
        norm_module = norm_module.cuda()

    return norm_module


def save_best_model(cfg, model, optimizer, epoch=0, best_metric=0.0):
    model_path = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_metric': best_metric
    }
    torch.save(state, model_path)
    # print(f"Saved best model to {model_path}")


def load_best_model(cfg, model):
    model_path = os.path.join(cfg.TRAIN.CHECKPOINT_DIR, "checkpoint_best.pth")
    if os.path.isfile(model_path):
        # print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")

        # support both bare state_dict and {'model_state': ...} formats
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
        msg = model.load_state_dict(state_dict, strict=True)
        assert set(msg.missing_keys) == set()

        # print(f"Loaded pre-trained model from {model_path}")
    # else:
        # print("=> no checkpoint found at '{}'".format(model_path))

    return model
