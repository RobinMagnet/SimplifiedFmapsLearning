import os
import sys
import time

from tqdm.auto import tqdm

import torch as th
import numpy as np

from diff_zo import diffZO

sys.path.append('../pyFM_dev/')
sys.path.append('../mydatasets/')
sys.path.append('../robin_utils/')
from pyFM.mesh import TriMesh
import pyFM.spectral as spectral
import pyFM.refine.zoomout as zoomout
from mydatasets import mydatasets as mydt
from robin_utils import utils as rbu


if __name__ == '__main__':
    torchdeviceId = "cuda:0"
    dataset = mydt.DT4DR_Dataset()

    res_dir = '/mnt/disk1/robin/Results/DiffZOTest/'

    os.makedirs(res_dir, exist_ok=True)

    os.makedirs(os.path.join(res_dir, 'zo'), exist_ok=True)
    os.makedirs(os.path.join(res_dir, 'zo_th'), exist_ok=True)
    os.makedirs(os.path.join(res_dir, 'zo_p'), exist_ok=True)
    os.makedirs(os.path.join(res_dir, 'zo_d'), exist_ok=True)
    os.makedirs(os.path.join(res_dir, 'zo_ot'), exist_ok=True)

    avg_np_time = 0
    avg_th_time = 0
    avg_th_time_precise = 0
    avg_th_time_d = 0
    avg_th_time_ot=0
    for pairind in tqdm(range(dataset.n_pairs)):
        pair_info = dataset.get_pair(pairind)

        mesh1 = TriMesh(pair_info['path2'], area_normalize=True).process(k=150, intrinsic=True)
        mesh2 = TriMesh(pair_info['path1'], area_normalize=True).process(k=150, intrinsic=True)


        evects1 = rbu.toTH(mesh1.eigenvectors).to(device=torchdeviceId)
        evects2 = rbu.toTH(mesh2.eigenvectors).to(device=torchdeviceId)
        mass2   = rbu.toTH(mesh2.vertex_areas).to(device=torchdeviceId)
        faces2 = rbu.toTH(mesh2.faces).to(device=torchdeviceId)
        faces1 = rbu.toTH(mesh1.faces).to(device=torchdeviceId)

        FM_12_init = rbu.toTH(spectral.mesh_p2p_to_FM(pair_info['p2p_12'], mesh1, mesh2, dims=15)).to(device=torchdeviceId)

        F1 = evects1[:,:FM_12_init.shape[1]] @ FM_12_init.T
        F2 = evects2[:,:FM_12_init.shape[0]]

        # start_time_np = time.time()
        # res_zo_np = zoomout.mesh_zoomout_refine(rbu.toNP(FM_12_init), mesh1, mesh2, nit=15, step=5, return_p2p=True, n_jobs=10, verbose=False)
        # avg_np_time += (time.time()-start_time_np) / dataset.n_pairs


        # zo_model = diffZO.KernelZoomOut(nn_only=True, precise=False, init_blur=1e-1, blur=1e-3, nit=15, step=5, k_init=20)
        # start_time_th = time.time()
        # res_zo_th = zo_model(F1, F2, evects1, evects2, mass2, faces1=faces1)
        # avg_th_time += (time.time()-start_time_th) / dataset.n_pairs

        # zo_model = diffZO.KernelZoomOut(nn_only=True, precise=True, init_blur=1e-1, blur=1e-3, nit=15, step=5, k_init=20)
        # start_time_th_p = time.time()
        # res_zo_th_precise = zo_model(F1, F2, evects1, evects2, mass2, faces1=faces1)
        # avg_th_time_precise += (time.time()-start_time_th_p) / dataset.n_pairs


        # zo_model = diffZO.KernelZoomOut(nn_only=False, precise=False, init_blur=1e-1, blur=1e-3, nit=15, step=5, k_init=20)
        # start_time_th_d = time.time()
        # res_zo_th_diff = zo_model(F1, F2, evects1, evects2, mass2, faces1=faces1)
        # avg_th_time_d += (time.time()-start_time_th_d) / dataset.n_pairs


        zo_model = diffZO.OTZoomOut(nn_only=False, precise=False, init_blur=1e-3, blur=1e-3, normalize=True, nit=15, step=5, k_init=10,
                           scaling=.7, n_inner=1)
        start_time_th_ot = time.time()
        res_zo_th_ot = zo_model(F1, F2, evects1, evects2, mass2)
        avg_th_time_ot += (time.time()-start_time_th_ot) / dataset.n_pairs

        # p2p_zo_np = res_zo_np[1]
        # p2p_zo_th = rbu.toNP(res_zo_th[1].get_nn())
        # p2p_zo_th_precise = rbu.toNP(res_zo_th_precise[1].get_nn())
        # p2p_zo_th_d = rbu.toNP(res_zo_th_diff[1].get_nn())
        p2p_zo_th_ot = rbu.toNP(res_zo_th_ot[1].get_nn())

        # rbu.save_ints(os.path.join(res_dir, 'zo', f'p2p_12_{pairind}'), p2p_zo_np)
        # rbu.save_ints(os.path.join(res_dir, 'zo_th', f'p2p_12_{pairind}'), p2p_zo_th)
        # rbu.save_ints(os.path.join(res_dir, 'zo_p', f'p2p_12_{pairind}'), p2p_zo_th_precise)
        # rbu.save_ints(os.path.join(res_dir, 'zo_d', f'p2p_12_{pairind}'), p2p_zo_th_d)
        rbu.save_ints(os.path.join(res_dir, 'zo_ot', f'p2p_12_{pairind}'), p2p_zo_th_d)
    
    print(f'Avg time zo : {avg_np_time}')
    print(f'Avg time zo_th : {avg_th_time}')
    print(f'Avg time zo_p : {avg_th_time_precise}')
    print(f'Avg time zo_d : {avg_th_time_d}')