import os
import os.path as osp
import sys
import numpy as np
import scipy.linalg
from tqdm import tqdm

# ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

from .misc import KNNSearch

# https://github.com/RobinMagnet/pyFM


def FM_to_p2p(FM, eigvects1, eigvects2):

    k2, k1 = FM.shape

    assert k1 <= eigvects1.shape[1], \
        f'At least {k1} should be provided, here only {eigvects1.shape[1]} are given'
    assert k2 <= eigvects2.shape[1], \
        f'At least {k2} should be provided, here only {eigvects2.shape[1]} are given'

    tree = KNNSearch(eigvects1[:, :k1] @ FM.T)
    matches = tree.query(eigvects2[:, :k2], k=1).flatten()

    return matches


def p2p_to_FM(p2p, eigvects1, eigvects2, A2=None):
    if A2 is not None:
        if A2.shape[0] != eigvects2.shape[0]:
            raise ValueError("Can't compute pseudo inverse with subsampled eigenvectors")

        if len(A2.shape) == 1:
            return eigvects2.T @ (A2[:, None] * eigvects1[p2p, :])

        return eigvects2.T @ A2 @ eigvects1[p2p, :]

    return scipy.linalg.lstsq(eigvects2, eigvects1[p2p, :])[0]


def zoomout_iteration(eigvects1, eigvects2, FM, step=1, A2=None):
    k2, k1 = FM.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step
    new_k1, new_k2 = k1 + step1, k2 + step2

    p2p = FM_to_p2p(FM, eigvects1, eigvects2)
    FM_zo = p2p_to_FM(p2p, eigvects1[:, :new_k1], eigvects2[:, :new_k2], A2=A2)

    return FM_zo


def zoomout_refine(eigvects1,
                   eigvects2,
                   FM,
                   nit=10,
                   step=1,
                   A2=None,
                   subsample=None,
                   return_p2p=False,
                   verbose=False):
    k2_0, k1_0 = FM.shape
    try:
        step1, step2 = step
    except TypeError:
        step1 = step
        step2 = step

    assert k1_0 + nit*step1 <= eigvects1.shape[1], \
        f"Not enough eigenvectors on source : \
        {k1_0 + nit*step1} are needed when {eigvects1.shape[1]} are provided"
    assert k2_0 + nit*step2 <= eigvects2.shape[1], \
        f"Not enough eigenvectors on target : \
        {k2_0 + nit*step2} are needed when {eigvects2.shape[1]} are provided"

    use_subsample = False
    if subsample is not None:
        use_subsample = True
        sub1, sub2 = subsample

    FM_zo = FM.copy()

    iterable = range(nit) if not verbose else tqdm(range(nit))
    for it in iterable:

        if use_subsample:
            FM_zo = zoomout_iteration(eigvects1[sub1], eigvects2[sub2], FM_zo, A2=None, step=step)

        else:
            FM_zo = zoomout_iteration(eigvects1, eigvects2, FM_zo, A2=A2, step=step)

    if return_p2p:
        p2p_zo = FM_to_p2p(FM_zo, eigvects1, eigvects2)
        return FM_zo, p2p_zo

    return FM_zo
