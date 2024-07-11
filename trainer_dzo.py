import os
import sys
# import random
import numpy as np
import torch as th
import wandb
import time
import glob
import shutil
import yaml
from omegaconf import OmegaConf
import importlib
from torch.utils.data import DataLoader, ConcatDataset
# from scipy.io import savemat
from tqdm.auto import tqdm
from pathlib import Path

# ROOT_DIR = osp.abspath(osp.dirname(__file__))
# if ROOT_DIR not in sys.path:
#     sys.path.append(ROOT_DIR)

from learn_zo.backbone.diffusionNet.layers import DiffusionNet
from learn_zo.backbone.FMNet import fmnet
# from learn_zo.models.attnfmaps import SpectralAttentionNet
# from learn_zo.models.utils import DiffNNSearch, FMAP_SOLVERS
from learn_zo.models.utils import to_numpy, validate_gradient, validate_tensor, fmap_to_image, bslice, diff_zoomout
from learn_zo.models.utils import orthogonality_s_loss
from learn_zo.data import get_data_dirs, collate_default, prepare_batch
from learn_zo.data.utils import load_geodist
from learn_zo.data import DATA_DIRS
# from learn_zo.data.utils import farthest_point_sampling
# from learn_zo.utils.fmap import FM_to_p2p
from learn_zo.utils.io import may_create_folder, save_pickle
from learn_zo.utils.convert import toNP
from learn_zo.utils.misc import incrange, validate_str, run_trainer, omegaconf_to_dotdict

from learn_zo.backbone.FMNet.fmnet import RegularizedFMNet

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DZO_dir = os.path.join(ROOT_DIR, 'DiffZo')
if DZO_dir not in sys.path:
    sys.path.append(DZO_dir)

from diff_zo import KernelZoomOut


# import pdb

class Trainer(object):
    STATE_KEYS = ['feat_model', 'dzo_layer', 'optimizer', 'scheduler']
    STATE_KEY_VERSION = 1

    def __init__(self, cfg):
        self.cfg = cfg

        self.supervised = self.cfg.get("supervised", False)
        if self.supervised:
            print('Supervised training')

        self.device = th.device(f'cuda:{cfg["gpu"]}' if th.cuda.is_available() else 'cpu')

        # self.spectral_dims = incrange(cfg['loss.spectral_dim'], cfg['loss.max_spectral_dim'], cfg['loss.spectral_step_size'])
        # print("do: ", cfg.feat_model.get('dropout_prob', 0.5))
        self.feat_model = DiffusionNet(C_in=cfg.feat_model.in_channels,
                                       C_out=cfg.feat_model.out_channels,
                                       C_width=cfg.feat_model.block_width,
                                       N_block=cfg.feat_model.num_blocks,
                                       dropout=cfg.feat_model.dropout,
                                       dropout_prob=cfg.feat_model.get('dropout_prob', 0.5),
                                       num_eigenbasis=cfg.feat_model.num_eigenbasis)
        # print(cfg.dzo_layer)
        self.dzo_layer = KernelZoomOut(**cfg.dzo_layer)

        if self.cfg.get('FM_layer', None) is not None:
            self.FM_layer = RegularizedFMNet(**cfg.FM_layer)
            self.STATE_KEYS += ['FM_layer']
        else:
            self.FM_layer = None


        self.feat_model = self.feat_model.to(self.device)
        self.dzo_layer = self.dzo_layer.to(self.device)

        self.spectral_dims = [self.dzo_layer.k_init, self.dzo_layer.k_final]

        self.dataloaders = dict()

        self.use_geod_val = cfg['data']['val']['use_geod']
        self.use_geod_test = cfg['data']['test']['use_geod']

        self.best_val_loss = np.inf

        self.use_smoothing = cfg.feat_model.smooth_features
        self.use_bij = cfg.loss.get("w_bij", 0) > 0
        self.normalize_feat = cfg.feat_model.get("normalize_features", False)
        self.use_consist_refine = cfg.loss.get("w_consist_refine", 0) > 0

    def _init_train(self, phase='train'):
        cfg = self.cfg

        exp_time = time.strftime('%y-%m-%d_%H-%M-%S')
        cfg['log_dir'] = cfg['log_dir'] + f'_{exp_time}'
        may_create_folder(cfg['log_dir'])

        self.start_epoch = 1

        parameters = list(self.feat_model.parameters())
        parameters += list(self.dzo_layer.parameters())

        self.optimizer = th.optim.Adam(parameters, lr=cfg.optim.lr, betas=(0.9, 0.99))

        if cfg.optim.get("scheduler_type", 'StepLR') == 'StepLR':
            self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer,
                                                        step_size=cfg.optim.decay_step,
                                                        gamma=cfg.optim.decay_gamma)
        elif cfg.optim.get("scheduler_type", 'StepLR') == 'CosineAnnealingLR':
            self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.optim.T_max, eta_min=cfg.optim.eta_min)

        dsets = list()
        for data_name in cfg.data.train.types:
            shape_cls = getattr(importlib.import_module(f'learn_zo.data.{data_name}'), 'ShapeDataset')
            pair_cls = getattr(importlib.import_module(f'learn_zo.data.{data_name}'), 'ShapePairDataset')
            shape_dir, cache_dir, corr_dir = get_data_dirs(cfg.data.root, data_name, phase)
            dset = shape_cls(shape_dir=shape_dir,
                             cache_dir=cache_dir,
                             mode=phase,
                             aug_noise_type=cfg.data.train.noise_type,
                             aug_noise_args=cfg.data.train.noise_args,
                             aug_rotation_type=cfg.data.train.rotation_type,
                             aug_rotation_args=cfg.data.train.rotation_args,
                             aug_scaling=cfg.data.train.scaling,
                             aug_scaling_args=cfg.data.train.scaling_args,
                             laplacian_type=cfg.data.laplacian_type,
                             feature_type=cfg.data.feature_type,
                             scale=cfg.data.train.get('scale', True))
            dset = pair_cls(corr_dir=corr_dir,
                            mode=phase,
                            num_corrs=cfg.data.num_corrs,
                            use_geodists=False,
                            fmap_sizes=self.spectral_dims,
                            shape_data=dset,
                            corr_loader=None)
            dsets.append(dset)

        dsets = ConcatDataset(dsets)
        dloader = DataLoader(dsets,
                             collate_fn=collate_default,
                             batch_size=cfg.data.train.batch_size,
                             shuffle=True,
                             num_workers=cfg.data.num_workers,
                             pin_memory=False,
                             drop_last=False)
        self.dataloaders[phase] = dloader

        wandb.init(project=cfg['project'],
                   dir=cfg['log_dir'],
                   group=cfg['group'],
                   notes=exp_time,
                   tags=[exp_time[:8]] + list(cfg.data.train.types),
                   settings=wandb.Settings(start_method='fork'))
        # print(cfg)
        wandb.config.update(omegaconf_to_dotdict(cfg.copy()))

        with open(os.path.join(cfg['log_dir'], 'config.yml'), 'w') as fh:
            # yaml.dump(cfg, fh)
            OmegaConf.save(cfg, fh)

        with open(os.path.join(cfg['log_dir'], 'model.txt'), 'w') as fh:
            fh.write(str(self.feat_model))
            fh.write('\n')
            fh.write(str(self.dzo_layer))
            if self.FM_layer is not None:
                fh.write('\n')
                fh.write(str(self.FM_layer))
            # fh.write('\n')
            # fh.write(str(self.nnsearcher))

        # code_backup_dir = os.path.join(wandb.run.dir, 'code_src')
        # for fileext in ['.py', '.yml', '.sh', 'Dockerfile']:
        #     for filepath in glob.glob('**/*{}'.format(fileext), recursive=True):
        #         may_create_folder(os.path.join(code_backup_dir, Path(filepath).parent))
        #         shutil.copy(os.path.join(ROOT_DIR, filepath), os.path.join(code_backup_dir, filepath))

        if validate_str(cfg['train_ckpt']):
            self._load_ckpt(cfg['train_ckpt'])

        self.temp_scheduler = self.cfg.get("dzo_scheduler", None)

    def _init_test(self, phase='test'):
        cfg = self.cfg

        assert validate_str(cfg['test_ckpt'])
        cfg['log_dir'] = str(Path(cfg['test_ckpt']).parent)
        may_create_folder(cfg['log_dir'])



        for data_name in cfg.data.test.types:
            shape_cls = getattr(importlib.import_module(f'learn_zo.data.{data_name}'), 'ShapeDataset')
            pair_cls = getattr(importlib.import_module(f'learn_zo.data.{data_name}'), 'ShapePairDataset')
            shape_dir, cache_dir, corr_dir = get_data_dirs(cfg.data.root, data_name, phase)
            if self.use_geod_test:
                geod_dir = os.path.join(cfg.data.root, DATA_DIRS[data_name], 'geodist')
            else:
                geod_dir = None
            dset = shape_cls(shape_dir=shape_dir,
                             cache_dir=cache_dir,
                             mode=phase,
                             aug_noise_type=None,
                             aug_noise_args=None,
                             aug_rotation_type=None,
                             aug_rotation_args=None,
                             aug_scaling=False,
                             aug_scaling_args=None,
                             laplacian_type=cfg.data.laplacian_type,
                             feature_type=cfg.data.feature_type,
                             geod_dir=geod_dir,
                             geod_in_loader=cfg["data"]["test"]["geod_in_loader"],
                             scale=cfg.data.test.get('scale', True))
            dset = pair_cls(corr_dir=corr_dir,
                            mode=phase,
                            num_corrs=cfg.data.num_corrs,
                            use_geodists=False,
                            fmap_sizes=self.spectral_dims,
                            shape_data=dset,
                            corr_loader=None)
            dloader = DataLoader(
                dset,
                collate_fn=collate_default,
                batch_size=cfg.data.test.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=False,
                drop_last=False,
            )
            self.dataloaders[f'{phase}_{data_name}'] = dloader

        self._load_ckpt(cfg['test_ckpt'])


    def _init_val(self, phase='val'):
        cfg = self.cfg

        # cfg['log_dir'] = str(Path(cfg['test_ckpt']).parent)
        # may_create_folder(cfg['log_dir'])

        for data_name in cfg.data.val.types:
            shape_cls = getattr(importlib.import_module(f'learn_zo.data.{data_name}'), 'ShapeDataset')
            pair_cls = getattr(importlib.import_module(f'learn_zo.data.{data_name}'), 'ShapePairDataset')
            shape_dir, cache_dir, corr_dir = get_data_dirs(cfg.data.root, data_name, phase) # Phase is useless here
            if self.use_geod_val:
                geod_dir = os.path.join(cfg.data.root, DATA_DIRS[data_name], 'geodist')
            else:
                geod_dir = None
            dset = shape_cls(shape_dir=shape_dir,
                             cache_dir=cache_dir,
                             mode='test',
                             aug_noise_type=None,
                             aug_noise_args=None,
                             aug_rotation_type=None,
                             aug_rotation_args=None,
                             aug_scaling=False,
                             aug_scaling_args=None,
                             laplacian_type=cfg['data']['laplacian_type'],
                             feature_type=cfg['data']['feature_type'],
                             geod_dir=geod_dir,
                             geod_in_loader=cfg["data"]["val"]["geod_in_loader"],
                             scale=cfg.data.val.get('scale', True))
            dset = pair_cls(corr_dir=corr_dir,
                            mode='test',
                            num_corrs=cfg['data']['num_corrs'],
                            use_geodists=False,
                            fmap_sizes=self.spectral_dims,
                            shape_data=dset,
                            corr_loader=None)
            dloader = DataLoader(
                dset,
                collate_fn=collate_default,
                batch_size=cfg.data.val.batch_size,
                shuffle=False,
                num_workers=cfg.data.num_workers,
                pin_memory=False,
                drop_last=False,
            )
            # self.dataloaders[f'val_{data_name}'] = dloader
            self.dataloaders[phase] = dloader

        # self._load_ckpt(cfg['test_ckpt'])
    # def get_loss(self, config):
    #     pass

    def compute_C12(self, F1, F2, evecs1, evecs2, mass2, K):
        # self.dzo_layer(feats0.squeeze(0), feats1.squeeze(0), batch_data['evecs0'][0], batch_data['evecs1'][0], batch_data['mass1'][0])

        T21_init = self.dzo_layer.compute_init(F1, F2)

        C12 = self.dzo_layer.compute_C12(T21_init, K, evecs1, evecs2, mass2)

        return C12

    def compute_loss(self, batch, preds):
        if self.supervised:
            C01_init = preds['C01_init']  # (B, K2_init, K1_init)
            C01 = preds['C01']  # (B, K2, K1)
            T10 = preds['T10']

            C01_gt = batch[f'fmap01_{C01.shape[-1]}_gt']  # (B, K2, K1)
            K1, K2 = C01.shape[-1], C01.shape[-2]
            K1_init, K2_init = C01_init.shape[-1], C01_init.shape[-2]

            loss = 1e2 * th.linalg.norm(C01_gt - C01) ** 2 / (K1*K2)
            loss += 1e2 * th.linalg.norm(C01_gt[..., :K2_init, :K1_init] - C01_init) ** 2 / (K1_init*K2_init)

        else:
            C01_init = preds['C01_init']  # (B, K2_init, K1_init)
            C01_final = preds['C01']  # (B, K2, K1)

            K2, K1 = C01_final.shape[-2], C01_final.shape[-1]
            K2_init, K1_init = C01_init.shape[-2], C01_init.shape[-1]

            loss = 0
            if self.cfg.loss.w_init > 0:
                loss += self.cfg.loss.w_init * 1e2 * orthogonality_s_loss(C01_init) / K1_init**2
            if self.cfg.loss.w_final > 0:
                loss += self.cfg.loss.w_final * 1e2 * orthogonality_s_loss(C01_final) / K1**2

            if self.cfg.loss.get('w_consist', 0) > 0:
                if not self.cfg.loss.get('consist_nodetach', False):
                    loss += self.cfg.loss.w_consist * 1e2 * th.linalg.norm(C01_init - C01_final[:,:K2_init, :K1_init].detach()) ** 2 / K1_init**2
                else:
                    loss += self.cfg.loss.w_consist * 1e2 * th.linalg.norm(C01_init - C01_final[:,:K2_init, :K1_init]) ** 2 / K1_init**2
            # loss = orthogonality_s_loss(C01_init)


            if self.cfg.loss.get("w_lap_bij_resolvant", 0) > 0:
                C01_init = preds['C01_init']  # (B, K2_init, K1_init)
                evals0 = batch["evals0"][...,:K1_init]  # (B, K1_init)
                evals1 = batch["evals1"][...,:K2_init]  # (B, K2_init)

                mask = fmnet.get_mask(evals0, evals1, 0.5) # (B, K2_init, K1_init)

                loss += self.cfg.loss.w_lap_bij_resolvant * 1e2 * th.linalg.norm(C01_init * mask) ** 2 / (K1_init*K2_init)


            if self.use_bij:
                C10_init = preds['C10_init']  # (B, K1_init, K2_init)
                C10_final = preds['C10']  # (B, K1, K2)

                bij1 = th.linalg.norm(C10_init @ C01_init - th.eye(K1_init, device=self.device)) ** 2 / K1_init**2
                bij2 = th.linalg.norm(C01_init @ C10_init - th.eye(K2_init, device=self.device)) ** 2 / K2_init**2

                loss += self.cfg.loss.w_bij * 1e2 * (bij1 + bij2)

                if self.cfg.loss.w_init > 0:
                    loss += self.cfg.loss.w_init * 1e2 * orthogonality_s_loss(C10_init) / K2_init**2

                if self.cfg.loss.w_final > 0:
                    loss += self.cfg.loss.w_final * 1e2 * orthogonality_s_loss(C10_final) / K2**2

                if self.cfg.loss.get('w_consist', 0) > 0:
                    if not self.cfg.loss.get('consist_nodetach', False):
                        loss += self.cfg.loss.w_consist * 1e2 * th.linalg.norm(C10_init - C10_final[:,:K1_init, :K2_init].detach()) ** 2 / K2_init**2
                    else:
                        loss += self.cfg.loss.w_consist * 1e2 * th.linalg.norm(C10_init - C10_final[:,:K1_init, :K2_init]) ** 2 / K2_init**2

        return loss

    def smooth(self, function, eigenvectors, mass):
        k = self.cfg.feat_model.k_smoothing
        # print(eigenvectors.shape, function.shape, mass.shape)
        return eigenvectors[...,:k] @ (eigenvectors[...,:k].mT @ (function * mass.unsqueeze(-1)))

    def _train_epoch(self, epoch, phase='train'):

        # print(self.dataloaders)
        cfg = self.cfg

        num_iters = len(self.dataloaders[phase])
        loader_iter = iter(self.dataloaders[phase])

        in_type = cfg.feat_model.in_type

        self.feat_model.train()
        self.dzo_layer.train()

        self.optimizer.zero_grad()
        for iter_idx in tqdm(range(num_iters), miniters=int(num_iters / 100), desc=f'Epoch: {epoch} {phase}'):
            global_step = (epoch - 1) * num_iters + (iter_idx + 1)
            log_dict = {'global_step': global_step}

            batch_data = next(loader_iter)
            batch_data = prepare_batch(batch_data, self.device)

            all_feats = [None, None]
            for pidx in range(2):
                feats = self.feat_model(x_in=batch_data[f'{in_type}{pidx}'].float(),
                                        mass=batch_data[f'mass{pidx}'].float(),
                                        L=None,
                                        evals=batch_data[f'evals{pidx}'].float(),
                                        evecs=batch_data[f'evecs{pidx}'].float(),
                                        gradX=batch_data[f'gradX{pidx}'],
                                        gradY=batch_data[f'gradY{pidx}'])
                all_feats[pidx] = feats.contiguous()
            feats0, feats1 = all_feats

            if self.use_smoothing:
                feats0 = self.smooth(feats0, batch_data['evecs0'], batch_data['mass0'])
                feats1 = self.smooth(feats1, batch_data['evecs1'], batch_data['mass1'])
                # print(feats0.shape, feats1.shape)

            if self.FM_layer is not None:
                K_est = self.dzo_layer.k_init
                evecs0_pinv = (batch_data['evecs0'][...,:K_est] * batch_data["mass0"].unsqueeze(-1)).mT  # (B, K, N1)
                evecs1_pinv = (batch_data['evecs1'][...,:K_est] * batch_data["mass1"].unsqueeze(-1)).mT  # (B, K, N2)
                C01_lin, C10_lin = self.FM_layer(feats0, feats1,
                                                 batch_data['evals0'][...,:K_est], batch_data['evals1'][...,:K_est],
                                                 evecs0_pinv, evecs1_pinv)

            if self.normalize_feat:
                    feats0 = feats0 / th.norm(feats0, dim=-1, keepdim=True)
                    feats1 = feats1 / th.norm(feats1, dim=-1, keepdim=True)

            assert feats0.shape[0] == feats1.shape[0] == 1, "batch size must be 1"
            [C01_init, C01], T10 = self.dzo_layer(feats0.squeeze(0), feats1.squeeze(0), batch_data['evecs0'][0], batch_data['evecs1'][0], batch_data['mass1'][0],
                                      return_init=True, return_T21=True)

            if self.use_consist_refine:
                K = C01.shape[-1]
                C01_pred_ref = self.compute_C12(feats0.squeeze(0), feats1.squeeze(0), batch_data['evecs0'][0], batch_data['evecs1'][0], batch_data['mass1'][0], K)

            if self.use_bij:
                [C10_init, C10], T01 = self.dzo_layer(feats1.squeeze(0), feats0.squeeze(0), batch_data['evecs1'][0], batch_data['evecs0'][0], batch_data['mass0'][0],
                                      return_init=True, return_T21=True)
                if self.use_consist_refine:
                    C10_pred_ref = self.compute_C12(feats1.squeeze(0), feats0.squeeze(0), batch_data['evecs1'][0], batch_data['evecs0'][0], batch_data['mass0'][0], K)

            preds = dict(feats0=feats0, feats1=feats1, C01_init=C01_init[None], C01=C01[None], T10=T10)

            if self.use_bij:
                preds.update(dict(C10_init=C10_init[None], C10=C10[None], T01=T01))

            if self.use_consist_refine:
                preds.update(dict(C01_pred_ref=C01_pred_ref[None]))
                if self.use_bij:
                    preds.update(dict(C10_pred_ref=C10_pred_ref[None]))

            if self.FM_layer is not None:
                preds.update(dict(C01_lin=C01_lin[None], C10_lin=C10_lin[None]))



            # loss_final = orthogonality_s_loss(C01_init[None])
            loss_final = self.compute_loss(batch_data, preds)

            loss = loss_final # + cfg['loss.interim_weight'] * loss_interim
            # pdb.set_trace()
            if (iter_idx + 1) % cfg['log_step'] == 0 or (iter_idx + 1) == num_iters:
                log_dict['loss'] = loss.item()
                log_dict['loss_final'] = loss_final.item()

                # log_dict['feats0'] = wandb.Histogram(toNP(feats0))
                # log_dict['feats1'] = wandb.Histogram(toNP(feats1))

                # nrow = int(math.ceil(math.sqrt(cfg['data.train.batch_size'])))
                nrow = np.ceil(np.sqrt(cfg.data.train.batch_size)).astype(int)
                # for sidx, SD in enumerate(spectral_dims):
                if validate_tensor(C01_init):
                    # log_dict[f'fmap01_pred_{C01_init.shape[0]}'] = wandb.Image(fmap_to_image(C01_init[None], nrow))
                    log_dict[f'fmap01_init'] = wandb.Image(fmap_to_image(C01_init[None], nrow))
                if validate_tensor(C01):
                    # log_dict[f'fmap01_pred_{C01.shape[0]}_dzo'] = wandb.Image(fmap_to_image(C01[None], nrow))
                    log_dict[f'fmap01_dzo'] = wandb.Image(fmap_to_image(C01[None], nrow))
                if validate_tensor(batch_data[f'fmap01_{C01.shape[0]}_gt']):
                    # log_dict[f'fmap01_gt_{C01.shape[0]}'] = wandb.Image(fmap_to_image(batch_data[f'fmap01_{C01.shape[-1]}_gt'], nrow))
                    log_dict[f'fmap01_gt'] = wandb.Image(fmap_to_image(batch_data[f'fmap01_{C01.shape[-1]}_gt'], nrow))
                wandb.log(log_dict)

            loss /= float(cfg.optim.accum_step)
            loss.backward()

            if (iter_idx + 1) % cfg.optim.accum_step == 0 or (iter_idx + 1) == num_iters:
                if validate_gradient(self.feat_model) and \
                   validate_gradient(self.dzo_layer) :
                    for m in [self.feat_model]:#, self.dzo_layer]:
                        th.nn.utils.clip_grad_value_(m.parameters(), cfg.optim.grad_clip)
                    self.optimizer.step()
                else:
                    print('[!] Invalid gradients')
                self.optimizer.zero_grad()

    def _val_epoch(self, epoch=None, phase='val'):
        # print(self.dataloaders)
        cfg = self.cfg
        num_iters = len(self.dataloaders[phase])
        dataset_fraction = cfg.data.val.get("fraction", 1)
        num_iters = int(num_iters * dataset_fraction)

        loader_iter = iter(self.dataloaders[phase])

        self.feat_model.eval()
        self.dzo_layer.eval()

        in_type = cfg.feat_model.in_type

        val_loss = 0
        val_loss_geod = 0
        val_coverage = 0
        for iter_idx in tqdm(range(num_iters), miniters=int(num_iters / 100), desc=phase, leave=False):
            global_step = iter_idx + 1

            batch_data = next(loader_iter)
            batch_data = prepare_batch(batch_data, self.device)

            if self.use_geod_val:
                # print(batch_data["geod_path0"])
                if "geodists0" in batch_data:
                    geodmat = batch_data["geodists0"][0]
                    sqrtarea = batch_data["sqrtarea0"][0]
                else:
                    geodmat, sqrtarea = load_geodist(batch_data["geod_path0"][0])

            with th.no_grad():
                all_feats = [None, None]
                for pidx in range(2):
                    feats = self.feat_model(x_in=batch_data[f'{in_type}{pidx}'].float(),
                                            mass=batch_data[f'mass{pidx}'].float(),
                                            L=None,
                                            evals=batch_data[f'evals{pidx}'].float(),
                                            evecs=batch_data[f'evecs{pidx}'].float(),
                                            gradX=batch_data[f'gradX{pidx}'],
                                            gradY=batch_data[f'gradY{pidx}'])
                    all_feats[pidx] = feats.contiguous()
                feats0, feats1 = all_feats

                if self.use_smoothing:
                    feats0 = self.smooth(feats0, batch_data['evecs0'], batch_data['mass0'])
                    feats1 = self.smooth(feats1, batch_data['evecs1'], batch_data['mass1'])
                    # print(feats0.shape, feats1.shape)
                if self.normalize_feat:
                    feats0 = feats0 / th.norm(feats0, dim=-1, keepdim=True)
                    feats1 = feats1 / th.norm(feats1, dim=-1, keepdim=True)

                [C01_init, C01], T10 = self.dzo_layer(feats0.squeeze(0), feats1.squeeze(0), batch_data['evecs0'][0], batch_data['evecs1'][0], batch_data['mass1'][0],
                                      return_init=True, return_T21=True)
                p2p_10 = T10.get_nn()


                C01_gt = batch_data[f'fmap01_{C01.shape[-1]}_gt'].squeeze(0)

                val_loss += th.linalg.norm(C01_gt - C01) ** 2

                if self.use_geod_val:
                    # sub_vts = batch["corr_gt"]
                    sub1 = batch_data["corr_gt"][0,:, 0].long()
                    sub2 = batch_data["corr_gt"][0,:, 1].long()
                    val_loss_geod += 1e2 * geodmat[(toNP(sub1), toNP(p2p_10[sub2]))].mean() / sqrtarea.sum()


                val_coverage += 1e2 * batch_data["mass0"][0][th.unique(p2p_10)].sum() / batch_data["mass0"].sum()

        val_coverage /= num_iters
        val_loss_geod /= num_iters
        val_loss /= num_iters
        log_dict = dict(val_loss=val_loss.item(), val_coverage=val_coverage.item())
        if self.use_geod_val:
            log_dict['val_loss_geod'] = val_loss_geod.item()

        wandb.log(log_dict)

        if self.use_geod_val:
            if val_loss_geod < self.best_val_loss:
                self.best_val_loss = val_loss_geod
                self._save_ckpt(epoch, f'{phase}_best')
        else:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_ckpt(epoch, f'{phase}_best')

        print(f'Current Val: {val_loss_geod.item():.4f}, Best Val: {self.best_val_loss.item():.4f}')

    def _test_epoch(self, epoch=None, phase='test'):
        cfg = self.cfg

        exp_time = time.strftime('%y-%m-%d_%H-%M-%S')
        out_root = os.path.join(cfg['log_dir'], f'{phase}_{exp_time}')
        may_create_folder(out_root)

        num_iters = len(self.dataloaders[phase])
        loader_iter = iter(self.dataloaders[phase])
        in_type = cfg.feat_model.in_type

        self.feat_model.eval()
        self.dzo_layer.eval()


        test_loss_geod = 0
        test_coverage = 0
        for iter_idx in tqdm(range(num_iters), miniters=int(num_iters / 100), desc=phase):
            global_step = iter_idx + 1

            batch_data = next(loader_iter)
            batch_data = prepare_batch(batch_data, self.device)

            if self.use_geod_test:
                # print(batch_data["geod_path0"])
                if "geodists0" in batch_data:
                    geodmat = batch_data["geodists0"][0]
                    sqrtarea = batch_data["sqrtarea0"][0]
                else:
                    geodmat, sqrtarea = load_geodist(batch_data["geod_path0"][0])


            with th.no_grad():
                all_feats = [None, None]
                for pidx in range(2):
                    feats = self.feat_model(x_in=batch_data[f'{in_type}{pidx}'].float(),
                                            mass=batch_data[f'mass{pidx}'].float(),
                                            L=None,
                                            evals=batch_data[f'evals{pidx}'].float(),
                                            evecs=batch_data[f'evecs{pidx}'].float(),
                                            gradX=batch_data[f'gradX{pidx}'],
                                            gradY=batch_data[f'gradY{pidx}'])
                    all_feats[pidx] = feats.contiguous()
                feats0, feats1 = all_feats
                if self.use_smoothing:
                    feats0 = self.smooth(feats0, batch_data['evecs0'], batch_data['mass0'])
                    feats1 = self.smooth(feats1, batch_data['evecs1'], batch_data['mass1'])
                    # print(feats0.shape, feats1.shape)
                if self.normalize_feat:
                    feats0 = feats0 / th.norm(feats0, dim=-1, keepdim=True)
                    feats1 = feats1 / th.norm(feats1, dim=-1, keepdim=True)

                [C01_init, C01], T10 = self.dzo_layer(feats0.squeeze(0), feats1.squeeze(0), batch_data['evecs0'][0], batch_data['evecs1'][0], batch_data['mass1'][0],
                                      return_init=True, return_T21=True)
                p2p_10 = T10.get_nn()

                # C01_gt = batch_data[f'fmap01_{C01.shape[-1]}_gt'].squeeze(0)

                if self.use_geod_val:
                    sub1 = batch_data["corr_gt"][0,:, 0]
                    sub2 = batch_data["corr_gt"][0,:, 1]
                    test_loss_geod += 1e2 * geodmat[(toNP(sub1), toNP(p2p_10[sub2.long()]))].mean() / sqrtarea.sum()


                test_coverage += 1e2 * batch_data["mass0"][0][th.unique(p2p_10)].sum() / batch_data["mass0"].sum()

            name0 = batch_data['name0'][0]
            name1 = batch_data['name1'][0]
            # fmap01_ref = to_numpy(torch.squeeze(fmap01_final, 0))
            # evecs0 = to_numpy(torch.squeeze(batch_data['evecs0'], 0))
            # evecs1 = to_numpy(torch.squeeze(batch_data['evecs1'], 0))

            # pmap10_ref = FM_to_p2p(fmap01_ref, evecs0, evecs1)

            to_save = {
                'id0': name0,
                'id1': name1,
                'pmap10_ref': p2p_10,
            }
            savepath = os.path.join(out_root, 'maps', f'{name0}-{name1}.p')
            may_create_folder(str(Path(savepath).parent))
            # savemat(matpath, to_save)
            save_pickle(savepath, to_save)


        test_coverage /= num_iters
        test_loss_geod /= num_iters
        print(f'{phase}: {test_loss_geod.item():.4f}')

        np.savetxt(os.path.join(out_root, 'test_values.txt'), [test_coverage.item(), test_loss_geod.item()])

    def end_of_epoch_update(self, epoch):
        if self.temp_scheduler is not None:
            decay = self.temp_scheduler.decay
            # self.dzo_layer.blur = self.dzo_layer.blur * decay
            self.dzo_layer.init_blur = self.dzo_layer.init_blur * decay

            # self.dzo_layer.blur = th.clamp(self.dzo_layer.blur, self.temp_scheduler.min_value, None)
            self.dzo_layer.init_blur = th.clamp(self.dzo_layer.init_blur, self.temp_scheduler.min_value, None)


        if OmegaConf.select(self.cfg, 'loss.schedulers') is not None:
            schedulers = self.cfg.loss.schedulers

            for lossname in schedulers.keys():
                if schedulers.get(lossname, None) is not None:
                    new_weight = self.cfg.loss[f'w_{lossname}'] * schedulers[lossname].decay
                    self.cfg.loss[f'w_{lossname}'] = np.clip(new_weight, schedulers[lossname].get("min_value", None), schedulers[lossname].get("max_value", None)).item()

    def train(self):
        cfg = self.cfg

        print('Start training')
        self._init_train()
        self._init_val()

        for epoch in range(self.start_epoch, cfg.data.train.epochs + 1):
            self._val_epoch(epoch)
            print(f'Epoch: {epoch}, LR = {self.scheduler.get_last_lr()}, Blur init = {self.dzo_layer.init_blur.item()}, Blur = {self.dzo_layer.blur.item()}')
            self._train_epoch(epoch)
            # exit(-1)
            self.scheduler.step()
            self.end_of_epoch_update(epoch)
            latest_ckpt_path = self._save_ckpt(epoch, 'latest')

            # self._init_val()
            # self._val_epoch(epoch)
        self._val_epoch(epoch)
        if os.path.isfile(os.path.join(cfg['log_dir'], 'ckpt_val_best.pth')):
            cfg['test_ckpt'] = os.path.join(cfg['log_dir'], 'ckpt_val_best.pth')
        else:
            print('Best CKPT not found, use latest CKPT')
            raise ValueError("Not a good idea")
            cfg['test_ckpt'] = latest_ckpt_path
        # cfg['test_ckpt'] = latest_ckpt_path
        print('Training finished')

    def test(self):
        cfg = self.cfg

        print('Start testing')
        self._init_test()

        for mode in self.dataloaders.keys():
            if mode.startswith('test'):
                self._test_epoch(phase=mode)

        print('Testing finished')

    def _save_ckpt(self, epoch, name=None):
        cfg = self.cfg

        state = {'epoch': epoch, 'version': self.STATE_KEY_VERSION}
        for k in self.STATE_KEYS:
            if hasattr(self, k):
                state[k] = getattr(self, k).state_dict()
        if name is None:
            filepath = os.path.join(cfg['log_dir'], f'ckpt_epoch_{epoch}.pth')
        else:
            filepath = os.path.join(cfg['log_dir'], f'ckpt_{name}.pth')
        th.save(state, filepath)
        print(f'Saved checkpoint to {filepath}')

        return filepath

    def _load_ckpt(self, filepath, keys=None):
        if keys is None:
            keys = self.STATE_KEYS
        if Path(filepath).is_file():
            state = th.load(filepath, map_location=self.device)
            if not 'version' in state or state['version'] != self.STATE_KEY_VERSION:
                raise RuntimeError(f'State version in checkpoint {filepath} does not match!')
            used_keys = list()
            for k in keys:
                if hasattr(self, k):
                    getattr(self, k).load_state_dict(state[k])
                    used_keys.append(k)
            if len(used_keys) == 0:
                raise RuntimeError(f'No state is loaded from checkpoint {filepath}!')
            print(f'Loaded checkpoint from {filepath} with keys {used_keys}')
        else:
            raise RuntimeError(f'Checkpoint {filepath} does not exist!')


if __name__ == '__main__':
    run_trainer(Trainer)
