import os
import os.path as osp
import sys
import itertools
import numpy as np
import torch as th
from torch.utils.data import Dataset
from pathlib import Path

from tqdm.auto import tqdm

# ROOT_DIR = osp.join(osp.abspath(osp.dirname(__file__)), '../')
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

import potpourri3d as pp3d
import learn_zo.utils.geometry as geom

from .utils import compute_vertex_normals
from .utils import compute_operators, load_operators
from .utils import farthest_point_sampling
from .utils import pmap_to_fmap, fmap_to_pmap
from .utils import compute_geodesic_distance
from .utils import load_geodist
from ..utils.io import list_files, may_create_folder


from ..utils.convert import toNP, toTH


class ShapeDataset(Dataset):
    TRAIN_IDX = np.arange(0, 80)
    TEST_IDX = np.arange(80, 100)

    def __init__(self,
                 shape_dir,
                 cache_dir,
                 mode,
                 aug_noise_type,
                 aug_noise_args,
                 aug_rotation_type,
                 aug_rotation_args,
                 aug_scaling,
                 aug_scaling_args,
                 num_eigenbasis=256,
                 laplacian_type='mesh',
                 feature_type=None,
                 geod_dir=None,
                 geod_in_loader=False,
                 scale=True,
                 normalize_desc=True,
                 **kwargs):
        """
        Initialize shape dataset.

        Parameters
        ----------
        shape_dir : str
            Path to the folder containing shape files.
        cache_dir : str
            Path to the folder containing cached operators.
        mode : str
            'train' or 'test'.
        aug_noise_type : str or None
            Type of noise augmentation.
        aug_noise_args : list or None
            Arguments for noise augmentation.
        aug_rotation_type : str or None
            Type of rotation augmentation.
        aug_rotation_args : list or None
            Arguments for rotation augmentation.
        aug_scaling : bool
            Whether to use scaling augmentation.
        aug_scaling_args : list or None
            Arguments for scaling augmentation.
        num_eigenbasis : int
            Number of eigenvectors to return.
        laplacian_type : str - 'mesh' or 'pcd'
            Type of Laplacian operator.
        feature_type : str or None
            Type of features to return. Different features are separated by '_'.
        geod_dir : str or None
            Path to the folder containing geodesic distances.
        """
        super().__init__()

        self.shape_dir = shape_dir
        self.cache_dir = cache_dir
        self.geod_dir = geod_dir
        self.geod_in_loader = geod_in_loader
        self.mode = mode
        self.aug_noise_type = aug_noise_type
        self.aug_noise_args = aug_noise_args
        self.aug_rotation_type = aug_rotation_type
        self.aug_rotation_args = aug_rotation_args
        self.aug_scaling = aug_scaling
        self.aug_scaling_args = aug_scaling_args
        self.num_eigenbasis = num_eigenbasis
        self.laplacian_type = laplacian_type
        self.feature_type = feature_type
        self.scale = scale
        self.normalize_desc = normalize_desc
        for k, w in kwargs.items():
            setattr(self, k, w)

        assert aug_noise_args is None or len(aug_noise_args) == 4
        assert aug_rotation_args is None or len(aug_rotation_args) == 3
        assert aug_scaling_args is None or len(aug_scaling_args) == 2

        print(f'Loading {mode} data from {shape_dir}')
        self.shape_list = self._get_file_list()
        self._prepare()

        self.randg = np.random.RandomState(0)

    def _get_file_list(self):
        file_list = list_files(self.shape_dir, '*.off', alphanum_sort=True)
        if self.mode.startswith('train'):
            assert self.TRAIN_IDX is not None
            shape_list = [file_list[idx] for idx in self.TRAIN_IDX]
        elif self.mode.startswith('test'):
            assert self.TEST_IDX is not None
            shape_list = [file_list[idx] for idx in self.TEST_IDX]
        else:
            raise RuntimeError(f'Mode {self.mode} is not supported.')
        return shape_list

    def _load_mesh(self, filepath, scale=True, return_vnormals=False):
        # import open3d as o3d

        # mesh = o3d.io.read_triangle_mesh(filepath)
        vertices, faces = pp3d.read_mesh(filepath)

        vertex_areas = geom.compute_vertex_areas(vertices, faces)
        center_mass = np.average(vertices, axis=0, weights=vertex_areas, keepdims=True)  # (1,3)
        vertices -= center_mass

        if scale:
            vertices /= np.sqrt(vertex_areas.sum())

        vertices = vertices.astype(np.float32)
        faces = faces.astype(np.int32)

        if return_vnormals:
            vnormals = geom.compute_vertex_normals(vertices, faces).astype(np.float32)
            return vertices, faces, vnormals
        else:
            return vertices, faces

    def _prepare(self):
        may_create_folder(self.cache_dir)
        for sid, sname in enumerate(tqdm(self.shape_list)):
            vertices, faces, vertex_normals = self._load_mesh(osp.join(self.shape_dir, sname),
                                                                       scale=self.scale,
                                                                       return_vnormals=True)

            cache_prefix = osp.join(self.cache_dir, f'{sname[:-4]}_{self.laplacian_type}_{self.num_eigenbasis}k')

            if self.scale:
                cache_path = cache_prefix + '_0n'
            else:
                cache_path = cache_prefix + '_0n_unscaled'
            
            if self.normalize_desc:
                cache_path += '_descnorm'
            
            cache_path += '.pt'

            if not Path(cache_path).is_file():
                if self.laplacian_type == 'mesh':
                    compute_operators(vertices, faces, vertex_normals, self.num_eigenbasis, cache_path, normalize_desc=self.normalize_desc)
                elif self.laplacian_type == 'pcd':
                    compute_operators(vertices, np.asarray([], dtype=np.int32), vertex_normals, self.num_eigenbasis,
                                      cache_path, normalize_desc=self.normalize_desc)
                else:
                    raise RuntimeError(f'Basis type {self.laplacian_type} is not supported')

            if self.aug_noise_type is not None and self.aug_noise_type != 'naive':
                raise ValueError('Didn\'t implement this yet')
                max_magnitude, max_levels = self.aug_noise_args[:2]
                randg = np.random.RandomState(sid)
                for nlevel in range(1, max_levels + 1):
                    cache_path = cache_prefix + f'_{nlevel}n.pt'
                    if Path(cache_path).is_file():
                        continue
                    noise_mag = max_magnitude * nlevel / max_levels
                    noise_mat = np.clip(noise_mag * randg.randn(vertices_np.shape[0], vertices_np.shape[1]), -noise_mag,
                                        noise_mag)
                    vertices_noise_np = vertices_np + noise_mat.astype(vertices_np.dtype)
                    vertex_normals_noise_np = compute_vertex_normals(vertices_noise_np, faces_np)

                    if self.laplacian_type == 'mesh':
                        compute_operators(vertices_noise_np, faces_np, vertex_normals_noise_np, self.num_eigenbasis, cache_path)
                    elif self.laplacian_type == 'pcd':
                        compute_operators(vertices_noise_np, np.asarray([], dtype=np.int32), vertex_normals_noise_np,
                                          self.num_eigenbasis, cache_path)
                    else:
                        raise RuntimeError(f'Basis type {self.laplacian_type} is not supported')

    def __getitem__(self, idx):
        sname = self.shape_list[idx]

        cache_prefix = osp.join(self.cache_dir, f'{sname[:-4]}_{self.laplacian_type}_{self.num_eigenbasis}k')
        if self.aug_noise_type is None:
            if self.scale:
                cache_path = cache_prefix + '_0n.pt' 
            else:
                cache_path = cache_prefix + '_0n_unscaled.pt'
        elif self.aug_noise_type == 'naive':
            if self.scale:
                cache_path = cache_prefix + '_0n.pt'
            else:
                cache_path = cache_prefix + '_0n_unscaled.pt'
            # cache_path = cache_prefix + '_0n.pt'
        elif self.aug_noise_type == 'random':
            raise ValueError('Didn\'t implement this yet')
            max_magnitude, max_levels = self.aug_noise_args[:2]
            cache_path = cache_prefix + '_{}n.pt'.format(self.randg.randint(0, max_levels + 1))
        else:
            raise ValueError('Didn\'t implement this yet')
            cache_path = cache_prefix + '_{self.aug_noise_type}.pt '

        assert Path(cache_path).is_file()

        sdict = load_operators(cache_path)
        sdict['idx'] = idx
        sdict['name'] = sname[:-4]

        if self.geod_dir is not None:
            sdict['geod_path'] = os.path.join(self.geod_dir, f'{sname[:-4]}.mat')

            if not Path(sdict['geod_path']).is_file():
                raise RuntimeError(f'Geodesic distance file {sdict["geod_path"]} does not exist.')
            
            if self.geod_in_loader:
                geodist, sqrtarea = load_geodist(sdict['geod_path'])
                sdict['geodists'] = geodist
                sdict['sqrtarea'] = sqrtarea

        

        sdict = self._centering(sdict)
        if self.aug_rotation_type is not None:
            sdict = self._random_rotation(sdict, self.randg, self.aug_rotation_type, self.aug_rotation_args)
        if self.aug_noise_type == 'naive':
            sdict = self._random_noise_naive(sdict, self.randg, self.aug_noise_args[2:])
        if self.aug_scaling:
            sdict = self._random_scaling(sdict, self.randg, self.aug_scaling_args)

        if self.feature_type is not None:
            # sdict['feats'] = np.concatenate([sdict[ft] for ft in self.feature_type.split('_')], axis=-1)
            sdict['feats'] = th.cat([sdict[ft] for ft in self.feature_type.split('_')], dim=-1)

        return sdict

    def __len__(self):
        return len(self.shape_list)

    def _centering(self, sdict):
        vertices = sdict['vertices']
        center = th.mean(vertices, dim=0, keepdims=True)
        sdict['vertices'] = vertices - center
        return sdict

    def _random_noise_naive(self, sdict, randg, args):
        vertices = sdict['vertices']
        # dtype = vertices.dtype
        shape = vertices.shape
        std, clip = args

        noise = th.from_numpy(np.clip(std * randg.randn(*shape), -clip, clip).astype(np.float32))
        sdict['vertices'] = vertices + noise.type_as(vertices)
        # astype(dtype)
        return sdict

    def _random_rotation(self, sdict, randg, axes, args):
        vertices = sdict['vertices']
        dtype = np.float32

        max_x, max_y, max_z = args
        if 'x' in axes:
            anglex = randg.rand() * max_x * np.pi / 180.0
            cosx = np.cos(anglex)
            sinx = np.sin(anglex)
            Rx = np.asarray([[1, 0, 0], [0, cosx, -sinx], [0, sinx, cosx]], dtype=dtype)
        else:
            Rx = np.eye(3, dtype=dtype)

        if 'y' in axes:
            angley = randg.rand() * max_y * np.pi / 180.0
            cosy = np.cos(angley)
            siny = np.sin(angley)
            Ry = np.asarray([[cosy, 0, siny], [0, 1, 0], [-siny, 0, cosy]], dtype=dtype)
        else:
            Ry = np.eye(3, dtype=dtype)

        if 'z' in axes:
            anglez = randg.rand() * max_z * np.pi / 180.0
            cosz = np.cos(anglez)
            sinz = np.sin(anglez)
            Rz = np.asarray([[cosz, -sinz, 0], [sinz, cosz, 0], [0, 0, 1]], dtype=dtype)
        else:
            Rz = np.eye(3, dtype=dtype)

        Rxyz = randg.permutation(np.stack((Rx, Ry, Rz), axis=0))
        R = Rxyz[2] @ Rxyz[1] @ Rxyz[0]
        sdict['vertices'] = vertices @ toTH(R.T).type_as(vertices)

        return sdict

    def _random_scaling(self, sdict, randg, args):
        scale_min, scale_max = args
        
        scale = scale_min + randg.rand(1, 3) * (scale_max - scale_min)

        vertices = sdict['vertices']
        sdict['vertices'] = vertices * toTH(scale)
        return sdict

    def get_name_id_map(self):
        return {sname[:-4]: sid for sid, sname in enumerate(self.shape_list)}


class ShapePairDataset(Dataset):

    def __init__(self, corr_dir, mode, num_corrs, use_geodists, fmap_sizes, shape_data, corr_loader, augment_fmap=False, **kwargs):
        """
        Dataset for training correspondence network.

        Parameters
        ----------
        corr_dir : str
            Path to the folder containing correspondence files.
        mode : str
            'train' or 'test'.
        num_corrs : int
            Number of correspondences to sample.
        use_geodists : bool or None
            Whether to use geodesic distances as additional input.
        fmap_sizes : list of int
            List of functional map sizes.
        shape_data : ShapeDataset
            Shape dataset.
        corr_loader : ShapePairDataset or None
            Correspondence dataset for loading initial correspondences.
        augment_fmap : bool
            Whether to augment functional maps with random scale or noise.
        kwargs : dict
            Other parameters.
        """
        super().__init__()
        self.corr_dir = corr_dir
        self.mode = mode
        self.num_corrs = num_corrs
        self.use_geodists = use_geodists
        self.augment_fmap = augment_fmap

        if np.issubdtype(type(fmap_sizes), np.integer):
            fmap_sizes = [fmap_sizes]
        self.fmap_sizes = fmap_sizes

        self.shape_data = shape_data
        self.corr_loader = corr_loader
        for k, w in kwargs.items():
            setattr(self, k, w)

        self._init()

        self.randg = np.random.RandomState(0)

    def _init(self):
        self.name_id_map = self.shape_data.get_name_id_map()
        self.pair_indices = list(itertools.combinations(range(len(self.shape_data)), 2))

    def __getitem__(self, idx):
        pidx = self.pair_indices[idx]
        sdict0 = self.shape_data[pidx[0]]
        sdict1 = self.shape_data[pidx[1]]
        return self._prepare_pair(sdict0, sdict1)

    def get_by_names(self, sname0, sname1):
        sdict0 = self.shape_data[self.name_id_map[sname0]]
        sdict1 = self.shape_data[self.name_id_map[sname1]]
        return self._prepare_pair(sdict0, sdict1)

    def _prepare_pair(self, sdict0, sdict1):
        pdict = dict()
        # Add keys with suffix 0 or 1
        for idx, sdict in enumerate([sdict0, sdict1]):
            for k in sdict.keys():
                pdict[f'{k}{idx}'] = sdict[k]

        is_train = self.mode.startswith('train')

        corr_gt = self._load_corr_gt(sdict0, sdict1)
        pdict['corr_gt'] = corr_gt

        for fmap_size in self.fmap_sizes:
            fmap01_gt = pmap_to_fmap(toNP(sdict0['evecs'][:, :fmap_size]), toNP(sdict1['evecs'][:, :fmap_size]), corr_gt)
            pdict[f'fmap01_{fmap_size}_gt'] = toTH(fmap01_gt)
        # return pdict

        skip_subsample = self.num_corrs is None or self.num_corrs == 0
        # If use subsample, compute farthest point sampling, geodesic distances, and functional maps
        if not skip_subsample:
            for idx in range(2):
                indices_sel = farthest_point_sampling(pdict[f'vertices{idx}'], self.num_corrs, random_start=is_train)
                for k in ['vertices', 'evecs', 'feats']:
                    kid = f'{k}{idx}'
                    if kid in pdict:
                        pdict[kid + '_sub'] = pdict[kid][indices_sel, :]
                if self.use_geodists:
                    geodists = compute_geodesic_distance(pdict[f'vertices{idx}'], pdict[f'faces{idx}'], indices_sel)
                    pdict[f'geodists{idx}_sub'] = toTH(geodists)
                pdict[f'vindices{idx}_sub'] = indices_sel
                assert th.is_tensor(indices_sel), f'Please convert indices_sel to tensor, not {type(indices_sel)}'

            fmap_size = self.fmap_sizes[-1]
            corr_gt_sub = toTH(fmap_to_pmap(pdict['evecs0_sub'][:, :fmap_size], pdict['evecs1_sub'][:, :fmap_size],
                                    pdict[f'fmap01_{fmap_size}_gt']))
            pdict['corr_gt_sub'] = corr_gt_sub

        if is_train and self.augment_fmap:
            fmap_size = self.fmap_sizes[0]
            axis = self.randg.choice([0, 1]).item()
            max_bases = fmap_size // 2
            noise_ratio = 0.5
            if self.randg.rand() > 0.5:
                pdict[f'fmap01_{fmap_size}'] = self._random_scale(pdict[f'fmap01_{fmap_size}_gt'], self.randg, axis, max_bases)
            else:
                pdict[f'fmap01_{fmap_size}'] = self._random_noise(pdict[f'fmap01_{fmap_size}_gt'], self.randg, axis, max_bases,
                                                                  noise_ratio)
        else:
            if self.corr_loader is not None:
                corr_init = self.corr_loader.get_by_names(sdict0['name'], sdict1['name'])
                assert corr_init.ndim == 2 and len(corr_init) == len(sdict1['vertices'])
                fmap_size = self.fmap_sizes[0]
                fmap01_init = toTH(pmap_to_fmap(toNP(sdict0['evecs'][:, :fmap_size]), toNP(sdict1['evecs'][:, :fmap_size]), corr_init))
                pdict[f'fmap01_{fmap_size}'] = fmap01_init
                pdict['pmap10'] = corr_init[:, 0]

        return pdict

    def _random_scale(self, fmap, randg, axis, max_bases):
        assert max_bases > 1
        assert axis in [0, 1]
        num_bases = randg.randint(1, max_bases)
        ids = randg.choice(fmap.shape[axis], num_bases, replace=False)
        fmap_out = fmap.clone()  # np.copy(fmap)
        if axis == 0:
            fmap_out[ids, :] *= toTH((randg.rand(num_bases, 1) * 2 - 1))
        else:
            fmap_out[:, ids] *= toTH((randg.rand(1, num_bases) * 2 - 1))
        return fmap_out

    def _random_noise(self, fmap, randg, axis, max_bases, max_ratio):
        assert max_bases > 1
        assert axis in [0, 1]
        num_bases = randg.randint(1, max_bases)
        ids = randg.choice(fmap.shape[axis], num_bases, replace=False)
        fmap_out = fmap.clone()  # np.copy(fmap)
        ratio = randg.rand() * max_ratio
        if axis == 0:
            maxvals = np.amax(np.abs(toNP(fmap_out[ids, :])), axis=1 - axis, keepdims=True)
            noise = ratio * maxvals * randg.randn(num_bases, fmap.shape[1 - axis])
            fmap_out[ids, :] += toTH(noise)
        else:
            maxvals = np.amax(np.abs(toNP(fmap_out[:, ids])), axis=1 - axis, keepdims=True)
            noise = ratio * maxvals * randg.randn(fmap.shape[1 - axis], num_bases)
            fmap_out[:, ids] += toTH(noise)
        return fmap_out

    def _load_corr_gt(self, sdict0, sdict1):
        corr0 = self._load_corr_file(sdict0['name'])
        corr1 = self._load_corr_file(sdict1['name'])
        corr_gt = np.stack((corr0, corr1), axis=1)
        return corr_gt

    def _load_corr_file(self, sname):
        corr_path = osp.join(self.corr_dir, f'{sname}.vts')
        corr = np.loadtxt(corr_path, dtype=np.int32)
        return corr - 1

    def __len__(self):
        return len(self.pair_indices)
