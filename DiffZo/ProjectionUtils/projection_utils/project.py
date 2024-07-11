import torch as th

from projection_utils.pc_to_mesh_proj import nn_query_precise_torch
from projection_utils.pc_to_mesh_proj_numpy import nn_query_precise_np


def nn_query_precise(vert_emb, faces, points_emb, return_dist=False, batch_size=None, verbose=False, clear_cache=True, n_jobs=1):

    if th.is_tensor(vert_emb):
        return nn_query_precise_torch(vert_emb, faces, points_emb, return_dist=return_dist, batch_size=batch_size, clear_cache=clear_cache)
    
    return nn_query_precise_np(vert_emb, faces, points_emb, return_dist=return_dist, batch_size=batch_size, verbose=verbose, n_jobs=n_jobs)
