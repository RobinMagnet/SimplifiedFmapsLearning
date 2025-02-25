
import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
import pykeops
import time

from einops import rearrange

from tqdm.auto import tqdm

from .maps import EmbPreciseMap, EmbP2PMap, KernelDistMap


class KernelZoomOut(nn.Module):
    def __init__(self, k_init=20, nit=2, step=10, blur=1e-2, init_blur=1, normalize=False,
                 nn_only=False, precise=False, n_inner=1):#, simple_init=False, init_blur=None):
        super().__init__()

        self.nit = nit
        self.k_init = k_init
        self.step = step
        self.n_inner = n_inner

        # self.blur = blur
        self.register_buffer('blur', th.Tensor([blur]))
        self.normalize=normalize

        # self.init_blur = init_blur
        self.register_buffer('init_blur', th.Tensor([init_blur]))
        self.nn_only=nn_only
        self.precise=precise

    @property
    def k_final(self):
        return self.k_init + self.nit * self.step

    def compute_C12(self, T21, k, evects1, evects2, mass2):
        """
        evects1 : (N1, K1) or (B,N1, K1)
        evects2 : (N2, K2) or (B,N2, K2)
        mass2   : (N2) or (B,N2)
        """
        # Linear algebra
        if np.issubdtype(type(k), np.integer):
            k1 = k
            k2 = k
        else:
            k1, k2 = k

        evects1_pb = T21.pull_back(evects1[:, :k1]) # ([B], N2, K1)

        return evects2[:,:k2].mT @ (mass2[..., None] * evects1_pb)

    def compute_T21(self, C12, evects1, evects2, blur=None, faces1=None):
        k2, k1 = C12.shape

        emb1 = evects1[:, :k1] @ C12.mT / k2
        emb2 = evects2[:, :k2] / k2

        emb1 = emb1.contiguous()
        emb2 = emb2.contiguous()

        blur = self.blur if blur is None else blur

        if self.nn_only:
            if self.precise:
                T21 = EmbPreciseMap(emb1, emb2, faces1)
            else:
                T21 = EmbP2PMap(emb1, emb2)
        else:
            T21 = KernelDistMap(emb1, emb2, blur=blur.item(), normalize=self.normalize)

        return T21

    def compute_init(self, F1, F2, faces1=None):
        if self.nn_only:
            if self.precise:
                T21 = EmbPreciseMap(F1, F2, faces1)
            else:
                T21 = EmbP2PMap(F1, F2)
        else:
            T21 = KernelDistMap(F1, F2, blur=self.init_blur.item(), normalize=self.normalize)

        return T21

    def forward(self, F1, F2, evects1, evects2, mass2, return_T21=True, return_init=False, faces1=None):
        # print(F1.shape, F2.shape, evects1.shape, evects2.shape, mass2.shape)

        # print(f' F1: {F1} \n F2: {F2} \n evects1: {evects1} \n evects2: {evects2} \n mass2: {mass2}')
        # print(f'F1 : {F1.is_contiguous()} \n F2 : {F2.is_contiguous()} \n evects1 : {evects1.is_contiguous()} \n evects2 : {evects2.is_contiguous()} \n mass2 : {mass2.is_contiguous()}')
        if self.nn_only:
            if self.precise:
                T21 = EmbPreciseMap(F1, F2, faces1)
            else:
                T21 = EmbP2PMap(F1, F2)
        else:
            T21 = KernelDistMap(F1, F2, blur=self.init_blur.item(), normalize=self.normalize)


        k_curr = self.k_init

        C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2)
        if return_init:
            C12_init = C12

        # C12_list.append(C12)

        # for i in tqdm(range(self.nit), leave=False):
        for i in range(self.nit):

            k_curr = k_curr + self.step
            for _ in range(self.n_inner):
                T21 = self.compute_T21(C12, evects1, evects2, faces1=faces1)

                C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2)

            # C12_list.append(C12)

        if not return_T21:
            if return_init:
                return [C12_init, C12]
            return C12

        T21 = self.compute_T21(C12, evects1, evects2, faces1=faces1)

        if return_init:
            return [C12_init, C12], T21
        return C12, T21


