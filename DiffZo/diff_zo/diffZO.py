
import os
import sys
import numpy as np
import torch as th
import torch.nn as nn
import pykeops
import time

from einops import rearrange

from tqdm.auto import tqdm

from .maps import EmbPreciseMap, EmbP2PMap, KernelDistMap, EmbOTMap


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


class OTZoomOut(KernelZoomOut):
    def __init__(self, k_init=20, nit=2, step=10, blur=1e-2, init_blur=1, normalize=False,
                 nn_only=False, precise=False,
                 scaling=.8, n_inner=1):#, simple_init=False, init_blur=None):
        super().__init__(k_init=k_init,
                         nit=nit, step=step, blur=blur, init_blur=init_blur, normalize=normalize,
                         nn_only=nn_only, precise=precise, n_inner=n_inner)

        self.scaling = scaling
        
    def compute_T21(self, C12, evects1, evects2, weights1=None, weights2=None, blur=None):
        k2, k1 = C12.shape

        emb1 = evects1[:, :k1] @ C12.mT / k2
        emb2 = evects2[:, :k2] / k2

        emb1 = emb1.contiguous()
        emb2 = emb2.contiguous()

        blur = self.blur if blur is None else blur


        T21 = EmbOTMap(emb1, emb2, 
                       weights1=weights1, weights2=weights2,
                       compute_diam=not self.normalize,
                       blur=blur, normalize=self.normalize,
                       scaling=self.scaling)

        return T21

    def forward(self, F1, F2, evects1, evects2, mass2, weights1=None, weights2=None, return_T21=True):

        T21 = EmbOTMap(F1, F2, 
                       weights1=weights1, weights2=weights2,
                       compute_diam=not self.normalize,
                       blur=self.init_blur, normalize=self.normalize,
                       scaling=self.scaling)


        k_curr = self.k_init

        C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2)

        # C12_list.append(C12)

        # for i in tqdm(range(self.nit), leave=False):
        for i in range(self.nit):

            k_curr = k_curr + self.step

            for _ in range(self.n_inner):
                
                T21 = self.compute_T21(C12, evects1, evects2, weights1=weights1, weights2=weights2)

                C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2)

            # C12_list.append(C12)

        if not return_T21:
            return C12

        T21 = self.compute_T21(C12, evects1, evects2, weights1=weights1, weights2=weights2)

        return C12, T21

# class DifferentiableZoomOut(nn.Module):
#     def __init__(self, k_init=20, nit=2, step=10, blur=1e-2, normalize=False):#, simple_init=False, init_blur=None):
#         super().__init__()

#         self.nit = nit
#         self.k_init = k_init
#         self.step = step

#         self.blur = blur
#         # self.


#     def compute_init(self, F1, F2, blur=None):
#         # return keops kernel

#         blur = self.blur if blur is None else blur

#         self.emb1 = F1 / F1.shape[1]
#         self.emb2 = F2 / F2.shape[1]

#         F_i, G_j = self.OT_solver(self.emb2, self.emb1)

#         return F_i, G_j, self.blur

#     def pull_back(self, f1, T21):
#         F_i, G_j, blur = T21

#         Y_j = self.emb1
#         X_i = self.emb2

#         k = X_i.shape[-1]

#         pb_dim = f1.shape[-1]

#         transfer = generic_sum(
#             "Exp( (F_i + G_j - IntInv(2)*SqDist(X_i,Y_j)) / E ) * L_j",  # See the formula above
#             f"Lab = Vi({pb_dim})",  # Output:  one vector of size 3 per line
#             "E   = Pm(1)",  # 1st arg: a scalar parameter, the temperature
#             f"X_i = Vi({k})",  # 2nd arg: one 2d-point per line
#             f"Y_j = Vj({k})",  # 3rd arg: one 2d-point per column
#             "F_i = Vi(1)",  # 4th arg: one scalar value per line
#             "G_j = Vj(1)",  # 5th arg: one scalar value per column
#             f"L_j = Vj({pb_dim})",
#         )  # 6th arg: one vector of size 3 per column

#         f1_pb = (
#             transfer(
#                 torch.Tensor([blur ** 2]).type(f1.type()),
#                 X_i,
#                 Y_j,
#                 F_i.view(-1, 1),
#                 G_j.view(-1, 1),
#                 f1,
#             )
#             / f1.shape[-2]
#         )

#         return f1_pb

#     def compute_C12(self, T21, k, evects1, evects2, mass2):
#         """
#         evects1 : (N1, K1) or (B,N1, K1)
#         evects2 : (N2, K2) or (B,N2, K2)
#         mass2   : (N2) or (B,N2)
#         """
#         # Linear algebra
#         if np.issubdtype(type(k), np.integer):
#             k1 = k
#             k2 = k
#         else:
#             k1, k2 = k

#         evects1_pb = self.pull_back(evects1[:,:k1], T21)  # ([B], N2, K1)

#         return evects2[:,:k2].mT @ (mass2[..., None] * evects1_pb)

#     def compute_T21(self, C12, evects1, evects2, blur=None):
#         k2, k1 = C12.shape

#         emb1 = evects1[:, :k1] @ C12.mT / k2
#         emb2 = evects2[:, :k2] / k2

#         self.emb1 = emb1.contiguous()
#         self.emb2 = emb2.contiguous()

#         blur = self.blur if blur is None else blur

#         # print(self.emb2.shape, self.emb1.shape)
#         # try:
#         F_i, G_j = self.OT_solver(self.emb2, self.emb1)

#         return F_i, G_j, blur

#     def forward(self, F1, F2, evects1, evects2, mass2, p2p_post_simple=False, return_T21=True):

#         C12_list = []

#         try:
#             T21 = self.base_T21(emb1=F1, emb2=F2, blur=self.init_blur) if self.simple_init else self.compute_init(F1, F2)
#             # print('check', T21[0].shape, T21[1].shape)
#         except ValueError:
#             print('init fail')
#             T21 = self.base_T21(emb1=F1, emb2=F2, blur=1e0)

#         k_curr = self.k_init

#         C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2)

#         C12_list.append(C12)

#         for i in range(self.nit):
#             k_curr += self.step

#             try:
#                 T21 = self.compute_T21(C12, evects1, evects2)
#             except ValueError:
#                 print('main fail')
#                 k2, k1 = C12.shape

#                 emb1 = evects1[:, :k1] @ C12.mT / k2
#                 emb2 = evects2[:, :k2] / k2

#                 self.emb1 = emb1.contiguous()
#                 self.emb2 = emb2.contiguous()

#                 T21 = self.base_T21(emb1=emb1, emb2=emb2, blur=self.blur)
#                 # C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2)
#                 # T21 = self.compute_T21(C12, evects1, evects2)

#             C12 = self.compute_C12(T21, k_curr, evects1, evects2, mass2)

#             C12_list.append(C12)

#         # print('end', F1.shape, F2.shape, C12.shape, T21[0].shape, T21[1].shape)
#         if not return_T21:
#             return C12_list

#         if p2p_post_simple:
#             k2, k1 = C12.shape

#             emb1 = evects1[:, :k1] @ C12.mT / k2
#             emb2 = evects2[:, :k2] / k2

#             self.emb1 = emb1.contiguous()
#             self.emb2 = emb2.contiguous()

#             T21 = self.base_T21(emb1=emb1, emb2=emb2, blur=self.blur)

#         else:
#             try:
#                 T21 = self.compute_T21(C12, evects1, evects2)
#             except ValueError:
#                 print('main fail')
#                 k2, k1 = C12.shape

#                 emb1 = evects1[:, :k1] @ C12.mT / k2
#                 emb2 = evects2[:, :k2] / k2

#                 self.emb1 = emb1.contiguous()
#                 self.emb2 = emb2.contiguous()

#                 T21 = self.base_T21(emb1=emb1, emb2=emb2, blur=self.blur)

#         return C12_list, T21

#     def compute_pointwise(self, T21):
#         F_i, G_j, blur = T21

#         Y_j = self.emb1
#         X_i = self.emb2

#         k = X_i.shape[-1]

#         formula = "Exp( (F_i + G_j - IntInv(2)*SqDist(X_i,Y_j)) / E )"  # See the formula above
#         variables = [  # Output:  one vector of size 3 per line
#             "E   = Pm(1)",  # 1st arg: a scalar parameter, the temperature
#             f"X_i = Vi({k})",  # 2nd arg: one 2d-point per line
#             f"Y_j = Vj({k})",  # 3rd arg: one 2d-point per column
#             "F_i = Vi(1)",  # 4th arg: one scalar value per line
#             "G_j = Vj(1)",  # 5th arg: one scalar value per column,
#             ]

#         nn_routine = Genred(formula, variables, reduction_op="ArgMax", axis=1)

#         matches = nn_routine(torch.Tensor([blur ** 2]).type(F_i.type()),
#                              X_i,
#                              Y_j,
#                              F_i.view(-1, 1),
#                              G_j.view(-1, 1),
#                              backend="auto").squeeze()

#         return matches

#     def base_T21(self, emb1=None, emb2=None, blur=None):
#         blur = self.blur if blur is None else blur

#         k = emb1.shape[1]

#         Y_j = self.emb1 if emb1 is None else emb1
#         X_i = self.emb2 if emb2 is None else emb2

#         self.emb1 = Y_j
#         self.emb2 = X_i

#         formula = "- IntInv(2)*SqDist(X_i,Y_j) / E"

#         variables = [  # Output:  one vector of size 3 per line
#             "E   = Pm(1)",  # 1st arg: a scalar parameter, the temperature
#             f"X_i = Vi({k})",  # 2nd arg: one 2d-point per line
#             f"Y_j = Vj({k})",  # 3rd arg: one 2d-point per column
#             ]

#         routine = Genred(formula, variables, reduction_op="LogSumExp", axis=1)

#         F_i = - blur**2 * routine(torch.Tensor([blur ** 2]).type(emb1.type()),
#                                   X_i,
#                                   Y_j,
#                                   backend="auto").squeeze()

#         G_j = torch.zeros(Y_j.shape[0]).type(F_i.type())

#         return F_i, G_j, blur