##########################################################################################
# Code is originally from the TAFAS (https://arxiv.org/pdf/2501.04970.pdf) implementation
# from https://github.com/kimanki/TAFAS by Kim et al. which is licensed under 
# Modified MIT License (Non-Commercial with Permission).
# You may obtain a copy of the License at
#
#    https://github.com/kimanki/TAFAS/blob/master/LICENSE
#
###########################################################################################

import random
from typing import Union
from pathlib import Path
import os


import numpy as np

import torch


def set_seeds(seed):
    if seed is None:
        return
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
def set_devices(devices: str):
    """
    Set cuda devices

    Args:
        devices (str): Comma separated string of device numbers. e.g., "0", "0, 1"
    """
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)


def mkdir(directory: Union[str, Path]):
    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    return directory


def prepare_inputs(inputs):
    # move data to the current GPU
    if isinstance(inputs, torch.Tensor):
        if(torch.cuda.is_available()):
            return inputs.float().cuda()
        else:
            return inputs.float()
    elif isinstance(inputs, (tuple, list)):
        return type(inputs)(prepare_inputs(v) for v in inputs)
