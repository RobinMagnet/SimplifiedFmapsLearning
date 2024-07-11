import sys
import os
# import os.path as osp
import numpy as np
import torch as th
from collections import defaultdict

from pathlib import Path

from ..utils.convert import sparse_th_from_args
# ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../')
# print(__file__)
ROOT_DIR = str(Path(__file__).parents[0])
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DATA_DIRS = {
    'faust': 'FAUST_r',
    'scape': 'SCAPE_r',
    'smalr': 'SMAL_r',
    'shrec19': 'SHREC_r',
    'dt4dintra': 'DT4D_r',
    'dt4dinter': 'DT4D_r',
}




def get_data_dirs(root: str, name: str, mode: str) -> tuple[str, str, str]:
    """
    Args:
        root: root directory
        name: dataset name
        mode: 'train' or 'test'
    Returns:
        shape_dir: directory of shapes
        cache_dir: directory of cached data
        corr_dir: directory of correspondences 
    """
    prefix = os.path.join(root, DATA_DIRS[name]) 
    shape_dir = os.path.join(prefix, 'shapes')
    cache_dir = os.path.join(prefix, 'cache_dzo')
    corr_dir = os.path.join(prefix, 'correspondences')
    return shape_dir, cache_dir, corr_dir


def collate_default(data_list: list[dict]) -> dict:
    """
    Args:
        data_list: list of dictionaries
    Returns:
        data_dict: dictionary of lists
    """
    data_dict = defaultdict(list)
    # Concatenate similar keys on a single dict into lists
    for pair_dict in data_list:
        for k, v in pair_dict.items():
            data_dict[k].append(v)

    # Stack numpy arrays in fmap, evals and _sub keys
    for k in data_dict.keys():
        if k.startswith('fmap') or k.startswith('evals') or k.endswith('_sub'):
            data_dict[k] = np.stack(data_dict[k], axis=0)
    
    # Check that all lists have the same batch size
    batch_size = len(data_list)
    for k, v in data_dict.items():
        assert len(v) == batch_size

    return data_dict


def prepare_batch(data_dict, device):
    """
    Args:
        data_dict: dictionary of lists
        device: torch.device
    Returns:
        data_dict: dictionary of tensors
    
    """
    # print('Data should be torch, modify this function')
    for k in data_dict.keys():
        # Transform np arry into torch tensor
        if isinstance(data_dict[k], np.ndarray):
            data_dict[k] = th.from_numpy(data_dict[k]).to(device)
        elif th.is_tensor(data_dict[k]):
            data_dict[k] = data_dict[k].to(device)
        else:
            if k.startswith('gradX') or \
               k.startswith('gradY') or \
               k.startswith('L'):
                tmp_list = [sparse_th_from_args(*st).to(device) for st in data_dict[k]]
                # from learn_zo.backbone.diffusionNet.utils import sparse_np_to_torch
                # tmp_list = [sparse_np_to_torch(st).to(device) for st in data_dict[k]]
                if len(data_dict[k]) == 1:
                    data_dict[k] = th.stack(tmp_list, dim=0)
                else:
                    data_dict[k] = tmp_list
            else:
                if isinstance(data_dict[k][0], np.ndarray):
                    tmp_list = [th.from_numpy(st).to(device) for st in data_dict[k]]
                    if len(data_dict[k]) == 1:
                        data_dict[k] = th.stack(tmp_list, dim=0).to(device)
                    else:
                        data_dict[k] = tmp_list
                elif th.is_tensor(data_dict[k][0]):
                    if len(data_dict[k]) == 1:
                        data_dict[k] = th.stack(data_dict[k], dim=0).to(device)
                    else:
                        data_dict[k] = data_dict[k].to(device)

    return data_dict
