import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.io import loadmat
import sys
import os
import os.path as osp
import math
import numpy as np
# import open3d as o3d
import potpourri3d as pp3d
import torch as th
from pathlib import Path

from einops import rearrange
import scipy.sparse as sparse

# ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

from learn_zo.utils import geometry as geom
import learn_zo.backbone.diffusionNet.geometry as diff_geom
from learn_zo.utils.convert import toNP, toTH, sparse_th_from_args
# from diffusion_net.utils import toNP

from learn_zo.utils.misc import KNNSearch
from learn_zo.utils.io import may_create_folder



# https://github.com/RobinMagnet/pyFM/blob/master/pyFM/signatures/HKS_functions.py
def HKS(evals, evects, time_list,scaled=False):
    """
    Returns the Heat Kernel Signature for num_T different values.
    The values of the time are interpolated in logscale between the limits
    given in the HKS paper. These limits only depends on the eigenvalues.

    Parameters
    ------------------------
    evals     : (K,) array of the K eigenvalues
    evecs     : (N,K) array with the K eigenvectors
    time_list : (num_T,) Time values to use
    scaled    : (bool) whether to scale for each time value

    Output
    ------------------------
    HKS : (N,num_T) array where each line is the HKS for a given t
    """
    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))  # (num_T,K)
    # weighted_evects = evects[None, :, :] * coefs[:, None,:]  # (num_T,N,K)
    # natural_HKS = np.einsum('tnk,nk->nt', weighted_evects, evects)
    natural_HKS = np.einsum('tk,nk,nk->nt', coefs, evects, evects)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_T)
        return (1/inv_scaling)[None,:] * natural_HKS

    return natural_HKS


def lm_HKS(evals, evects, landmarks, time_list, scaled=False):
    """
    Returns the Heat Kernel Signature for some landmarks and time values.


    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    landmarks   : (p,) indices of landmarks to compute
    time_list   : (num_T,) values of t to use

    Output
    ------------------------
    landmarks_HKS : (N,num_E*p) array where each column is the HKS for a given t for some landmark
    """

    evals_s = np.asarray(evals).flatten()
    t_list = np.asarray(time_list).flatten()

    coefs = np.exp(-np.outer(t_list, evals_s))  # (num_T,K)
    weighted_evects = evects[None, landmarks, :] * coefs[:,None,:]  # (num_T,p,K)

    landmarks_HKS = np.einsum('tpk,nk->ptn', weighted_evects, evects)  # (p,num_T,N)
    landmarks_HKS = np.einsum('tk,pk,nk->ptn', coefs, evects[landmarks, :], evects)  # (p,num_T,N)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_T,)
        landmarks_HKS = (1/inv_scaling)[None,:,None] * landmarks_HKS  # (p,num_T,N)

    return rearrange(landmarks_HKS, 'p T N -> N (p T)')


def auto_HKS(evals, evects, num_T, landmarks=None, scaled=True):
    """
    Compute HKS with an automatic choice of tile values

    Parameters
    ------------------------
    evals       : (K,) array of  K eigenvalues
    evects      : (N,K) array with K eigenvectors
    landmarks   : (p,) if not None, indices of landmarks to compute.
    num_T       : (int) number values of t to use
    Output
    ------------------------
    HKS or lm_HKS : (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    for some landmark
    """

    abs_ev = sorted(np.abs(evals))
    t_list = np.geomspace(4*np.log(10)/abs_ev[-1], 4*np.log(10)/abs_ev[1], num_T)

    if landmarks is None:
        return HKS(abs_ev, evects, t_list, scaled=scaled)
    else:
        return lm_HKS(abs_ev, evects, landmarks, t_list, scaled=scaled)


# https://github.com/RobinMagnet/pyFM/blob/master/pyFM/signatures/WKS_functions.py
def WKS(evals, evects, energy_list, sigma, scaled=False):
    """
    Returns the Wave Kernel Signature for some energy values.

    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    energy_list : (num_E,) values of e to use
    sigma       : (float) [positive] standard deviation to use
    scaled      : (bool) Whether to scale each energy level

    Output
    ------------------------
    WKS : (N,num_E) array where each column is the WKS for a given e
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-5)[0].flatten()
    evals = evals[indices]
    evects = evects[:, indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:,None] - np.log(np.abs(evals))[None,:])/(2*sigma**2))  # (num_E,K)

    # weighted_evects = evects[None, :, :] * coefs[:,None, :]  # (num_E,N,K)

    # natural_WKS = np.einsum('tnk,nk->nt', weighted_evects, evects)  # (N,num_E)
    natural_WKS = np.einsum('tk,nk,nk->nt', coefs, evects, evects)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E)
        return (1/inv_scaling)[None,:] * natural_WKS
    
    return natural_WKS


def lm_WKS(evals, evects, landmarks, energy_list, sigma, scaled=False):
    """
    Returns the Wave Kernel Signature for some landmarks and energy values.


    Parameters
    ------------------------
    evects      : (N,K) array with the K eigenvectors of the Laplace Beltrami operator
    evals       : (K,) array of the K corresponding eigenvalues
    landmarks   : (p,) indices of landmarks to compute
    energy_list : (num_E,) values of e to use
    sigma       : int - standard deviation

    Output
    ------------------------
    landmarks_WKS : (N,num_E*p) array where each column is the WKS for a given e for some landmark
    """
    assert sigma > 0, f"Sigma should be positive ! Given value : {sigma}"

    evals = np.asarray(evals).flatten()
    indices = np.where(evals > 1e-2)[0].flatten()
    evals = evals[indices]
    evects = evects[:,indices]

    e_list = np.asarray(energy_list)
    coefs = np.exp(-np.square(e_list[:, None] - np.log(np.abs(evals))[None, :]) / (2*sigma**2))  # (num_E,K)
    # weighted_evects = evects[None, landmarks, :] * coefs[:,None,:]  # (num_E,p,K)

    # landmarks_WKS = np.einsum('tpk,nk->ptn', weighted_evects, evects)  # (p,num_E,N)
    landmarks_WKS = np.einsum('tk,pk,nk->ptn', coefs, evects[landmarks, :], evects)  # (p,num_E,N)

    if scaled:
        inv_scaling = coefs.sum(1)  # (num_E,)
        landmarks_WKS = (1/inv_scaling)[None,:,None] * landmarks_WKS

    # return landmarks_WKS.reshape(-1,evects.shape[0]).T  # (N,p*num_E)
    return rearrange(landmarks_WKS, 'p T N -> N (p T)')


def auto_WKS(evals, evects, num_E, landmarks=None, scaled=True):
    """
    Compute WKS with an automatic choice of scale and energy

    Parameters
    ------------------------
    evals       : (K,) array of  K eigenvalues
    evects      : (N,K) array with K eigenvectors
    landmarks   : (p,) If not None, indices of landmarks to compute.
    num_E       : (int) number values of e to use
    Output
    ------------------------
    WKS or lm_WKS : (N,num_E) or (N,p*num_E)  array where each column is the WKS for a given e
                    and possibly for some landmarks
    """
    abs_ev = sorted(np.abs(evals))

    e_min,e_max = np.log(abs_ev[1]),np.log(abs_ev[-1])
    sigma = 7*(e_max-e_min)/num_E

    e_min += 2*sigma
    e_max -= 2*sigma

    energy_list = np.linspace(e_min,e_max,num_E)

    if landmarks is None:
        return WKS(abs_ev, evects, energy_list, sigma, scaled=scaled)
    else:
        return lm_WKS(abs_ev, evects, landmarks, energy_list, sigma, scaled=scaled)


def compute_hks(evecs, evals, mass, n_descr=100, subsample_step=5, n_eig=35, normalize=True):
    """
    Compute Heat Kernel Signature (HKS) descriptors.
    
    Args:
        evecs: (N, K) eigenvectors of the Laplace-Beltrami operator
        evals: (K,) eigenvalues of the Laplace-Beltrami operator
        mass: (N,) vertex masses
        n_descr: (int) number of descriptors
        subsample_step: (int) subsampling step
        n_eig: (int) number of eigenvectors to use
    
    Returns:
        feats: (N, n_descr) HKS descriptors
    """
    feats = auto_HKS(evals[:n_eig], evecs[:, :n_eig], n_descr, scaled=True)
    feats = feats[:, np.arange(0, feats.shape[1], subsample_step)]
    if normalize:
        feats_norm2 = np.einsum('np,n->p', feats**2, mass).flatten()
        feats /= np.sqrt(feats_norm2)[None, :]
    return feats.astype(np.float32)


def compute_wks(evecs, evals, mass, n_descr=100, subsample_step=5, n_eig=35, normalize=True):
    """
    Compute Wave Kernel Signature (WKS) descriptors.

    Args:
        evecs: (N, K) eigenvectors of the Laplace-Beltrami operator
        evals: (K,) eigenvalues of the Laplace-Beltrami operator
        mass: (N,) vertex masses
        n_descr: (int) number of descriptors
        subsample_step: (int) subsampling step
        n_eig: (int) number of eigenvectors to use
    
    Returns:
        feats: (N, n_descr) WKS descriptors
    """
    feats = auto_WKS(evals[:n_eig], evecs[:, :n_eig], n_descr, scaled=True)
    feats = feats[:, np.arange(0, feats.shape[1], subsample_step)]
    # print("wks_shape",feats.shape, mass.shape)
    if normalize:
        feats_norm2 = np.einsum('np,n->p', feats**2, mass).flatten()
        feats /= np.sqrt(feats_norm2)[None, :]
    # feats_norm2 = np.einsum('np,n->p', feats**2, mass).flatten()
    # feats /= np.sqrt(feats_norm2)[None, :]
    return feats.astype(np.float32)


def compute_geodesic_distance(V, F, vindices):
    """
    Compute geodesic distance from a set of vertices to all other vertices.

    Args:
        V: (N, 3) vertices
        F: (M, 3) faces
        vindices: (P,) vertex indices
    
    Returns:
        dists: (P, N) geodesic distances
    """
    solver = pp3d.MeshHeatMethodDistanceSolver(np.asarray(V, dtype=float), np.asarray(F, dtype=int))
    dists = [solver.compute_distance(vid)[vindices].astype(np.float32) for vid in vindices]
    dists = np.stack(dists, axis=0)
    assert dists.ndim == 2
    return dists.astype(np.float32)


# def compute_vertex_normals(vertices, faces):
#     mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
#     mesh.compute_vertex_normals()
#     return np.asarray(mesh.vertex_normals, dtype=np.float32)


def compute_vertex_normals(vertices, faces):
    """
    Compute per-vertex normals of a triangular mesh, weighted by the area of adjacent faces.

    Parameters
    -----------------------------
    vertices     : (n,3) array of vertices coordinates
    faces        : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    vert_areas : (n,) array of per-vertex areas
    """

    n_faces = faces.shape[0]
    n_vertices = vertices.shape[0]

    v1 = vertices[faces[:, 0]]  # (m,3)
    v2 = vertices[faces[:, 1]]  # (m,3)
    v3 = vertices[faces[:, 2]]  # (m,3)

    # That is 2* A(T) n(T) with A(T) area of face T
    face_normals_weighted = np.cross(1e3*(v2-v1), 1e3*(v3-v1))  # (m,3)

    # A simple version should be :
    # vert_normals = np.zeros((n_vertices,3))
    # np.add.at(vert_normals, faces.flatten(),np.repeat(face_normals_weighted,3,axis=0))
    # But this code is way faster in practice

    In = np.repeat(faces.flatten(), 3)  # (9m,)
    Jn = np.tile(np.arange(3), 3*n_faces)  # (9m,)
    Vn = np.tile(face_normals_weighted, (1,3)).flatten()  # (9m,)

    vert_normals = sparse.coo_matrix((Vn, (In, Jn)), shape=(n_vertices, 3))
    vert_normals = np.asarray(vert_normals.todense())
    vert_normals /= (1e-6 + np.linalg.norm(vert_normals, axis=1, keepdims=True))

    return vert_normals.astype(np.float32)


def compute_faces_areas(vertices, faces):
    """
    Compute per-face areas of a triangular mesh

    Parameters
    -----------------------------
    vertices : (n,3) array of vertices coordinates
    faces    : (m,3) array of vertex indices defining faces

    Output
    -----------------------------
    faces_areas : (m,) array of per-face areas
    """

    v1 = vertices[faces[:,0]]  # (m,3)
    v2 = vertices[faces[:,1]]  # (m,3)
    v3 = vertices[faces[:,2]]  # (m,3)
    faces_areas = 0.5 * np.linalg.norm(np.cross(v2-v1,v3-v1),axis=1)  # (m,)

    return faces_areas

def compute_surface_area(vertices, faces):
    faces_areas = compute_faces_areas(vertices, faces)
    return faces_areas.sum()
 

# def load_mesh(filepath, scale=True, return_vnormals=False):
#     mesh = o3d.io.read_triangle_mesh(filepath)

#     tmat = np.identity(4, dtype=np.float32)
#     center = mesh.get_center()
#     tmat[:3, 3] = -center
#     if scale:
#         smat = np.identity(4, dtype=np.float32)
#         area = mesh.get_surface_area()
#         smat[:3, :3] = np.identity(3, dtype=np.float32) / math.sqrt(area)
#         tmat = smat @ tmat
#     mesh.transform(tmat)

#     vertices = np.asarray(mesh.vertices, dtype=np.float32)
#     faces = np.asarray(mesh.triangles, dtype=np.int32)
#     if return_vnormals:
#         mesh.compute_vertex_normals()
#         vnormals = np.asarray(mesh.vertex_normals, dtype=np.float32)
#         return vertices, faces, vnormals
#     else:
#         return vertices, faces


# def save_mesh(filepath, vertices, faces):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(faces))
    o3d.io.write_triangle_mesh(filepath, mesh)


def load_operators(filepath):
    """
    Load surface operators (frames, mass, L, evals, evecs, gradX, gradY) and descriptors (HKS, WKS).
    Build sparse matrices from indices, values, and shape.
    """

    # npzfile = np.load(filepath, allow_pickle=False)
    cached_data = th.load(filepath, map_location='cpu')

    # def read_sp_mat(prefix):
    #     data = npzfile[prefix + '_data']
    #     indices = npzfile[prefix + '_indices']
    #     indptr = npzfile[prefix + '_indptr']
    #     shape = npzfile[prefix + '_shape']
    #     mat = scipy.sparse.csc_matrix((data, indices, indptr), shape=shape)
    #     return mat

    data = dict(
        vertices=cached_data['vertices'],
        faces=cached_data['faces'],
        frames=cached_data['frames'],
        mass=cached_data['mass'],
        # L=sparse_th_from_args(*cached_data['L']),
        L=cached_data['L'],
        evals=cached_data['evals'],
        evecs=cached_data['evecs'],
        # gradX=sparse_th_from_args(*cached_data['gradX']),
        # gradY=sparse_th_from_args(*cached_data['gradY']),
        gradX=cached_data['gradX'],
        gradY=cached_data['gradY'],
        hks=cached_data['hks'],
        wks=cached_data['wks'], 
    )

    return data


def compute_operators(verts, faces, normals, k_eig, cache_path=None,  normalize_desc=True):
    """
    Compute surface operators (frames, mass, L, evals, evecs, gradX, gradY) and descriptors (HKS, WKS).
    Sparse matrices are stored as indices, values, and shape.
    """
    verts = toTH(verts)
    faces = toTH(faces)
    frames, mass, L, evals, evecs, gradX, gradY = diff_geom.compute_operators(verts,
                                                                              faces,
                                                                              k_eig,
                                                                              normals=th.from_numpy(normals))
    
    
    L = (L.indices(), L.values(), L.shape)
    gradX = (gradX.indices(), gradX.values(), gradX.shape)
    gradY = (gradY.indices(), gradY.values(), gradY.shape)

    assert evecs.shape[-1] == k_eig

    hks = toTH(compute_hks(toNP(evecs), toNP(evals), toNP(mass), n_descr=128, subsample_step=1, n_eig=128, normalize=normalize_desc))
    wks = toTH(compute_wks(toNP(evecs), toNP(evals), toNP(mass), n_descr=128, subsample_step=1, n_eig=128, normalize=normalize_desc))


    all_data = dict(
        vertices=verts,
        faces=faces,
        frames=frames,
        mass=mass,
        L=L,
        evals=evals,
        evecs=evecs,
        gradX=gradX,
        gradY=gradY,
        hks=hks,
        wks=wks)
    
    if cache_path is not None and not Path(cache_path).is_file():
        may_create_folder(str(Path(cache_path).parent))
        th.save(
            all_data | dict(k_eig=k_eig),
            cache_path
        )

    return all_data


def load_geodist(filepath):
    data = loadmat(filepath)
    if 'geodist' in data and 'sqrt_area' in data:
        geodist = np.asarray(data['geodist'], dtype=np.float32)
        sqrt_area = data['sqrt_area'].toarray().flatten()[0]
    elif 'G' in data and 'SQRarea' in data:
        geodist = np.asarray(data['G'], dtype=np.float32)
        sqrt_area = data['SQRarea'].flatten()[0]
    else:
        raise RuntimeError(f'File {filepath} does not have geodesics data.')
    return geodist, sqrt_area


def farthest_point_sampling(points, max_points, random_start=True):
    import torch_cluster

    if th.is_tensor(points):
        device = points.device
        is_batch = points.dim() == 3
        if not is_batch:
            points = th.unsqueeze(points, dim=0)
        assert points.dim() == 3

        B, N, D = points.size()
        assert N >= max_points
        bindices = th.flatten(th.unsqueeze(th.arange(B), 1).repeat(1, N)).long().to(device)
        points = th.reshape(points, (B * N, D)).float()
        sindices = torch_cluster.fps(points, bindices, ratio=float(max_points) / N, random_start=random_start)
        if is_batch:
            sindices = th.reshape(sindices, (B, max_points)) - th.unsqueeze(th.arange(B), 1).long().to(device) * N
    elif isinstance(points, np.ndarray):
        device = th.device('cpu')
        is_batch = points.ndim == 3
        if not is_batch:
            points = np.expand_dims(points, axis=0)
        assert points.ndim == 3

        B, N, D = points.shape
        assert N >= max_points
        bindices = np.tile(np.expand_dims(np.arange(B), 1), (1, N)).flatten()
        bindices = th.as_tensor(bindices, device=device).long()
        points = th.as_tensor(np.reshape(points, (B * N, D)), device=device).float()
        sindices = torch_cluster.fps(points, bindices, ratio=float(max_points) / N, random_start=random_start)
        sindices = sindices.cpu().numpy()
        if is_batch:
            sindices = np.reshape(sindices, (B, max_points)) - np.expand_dims(np.arange(B), 1) * N
    else:
        raise NotImplementedError
    return sindices


def lstsq(A, B):
    assert A.ndim == B.ndim == 2
    sols = scipy.linalg.lstsq(A, B)[0]
    return sols


def pmap_to_fmap(evecs0, evecs1, corrs):
    assert evecs0.ndim == evecs1.ndim == corrs.ndim == 2

    evecs0_sel = evecs0[corrs[:, 0], :]
    evecs1_sel = evecs1[corrs[:, 1], :]
    fmap01_t = lstsq(evecs0_sel, evecs1_sel)
    return fmap01_t.T


def fmap_to_pmap(evecs0, evecs1, fmap01):
    assert fmap01.ndim == 2
    K1, K0 = fmap01.shape
    knnsearch = KNNSearch(evecs0[..., :K0] @ fmap01.T)
    pmap10 = knnsearch.query(evecs1[..., :K1], k=1)
    return np.stack((pmap10, np.arange(len(pmap10))), axis=1)
