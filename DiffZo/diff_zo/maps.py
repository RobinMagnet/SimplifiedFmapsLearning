import sys
import os
from pathlib import Path

import torch as th
import pykeops

from einops import rearrange


SOURCE_DIR = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(os.path.join(SOURCE_DIR, 'geomloss_mod'))
sys.path.append(os.path.join(SOURCE_DIR, 'ProjectionUtils'))
from geomloss_mod.samples_loss import SamplesLoss
from projection_utils import nn_query_precise

def nn_query(X, Y):
    formula = pykeops.torch.Genred('SqDist(X,Y)',
                    [f'X = Vi({X.shape[-1]})',          # First arg  is a parameter,    of dim 1
                    f'Y = Vj({Y.shape[-1]})',          # Second arg is indexed by "i", of dim
                    ],
                    reduction_op='ArgMin',
                    axis=0)
    
    return formula(X, Y).squeeze(-1)


class PointWiseMap:
    def __init__(self):
        pass
    
    def pull_back(self, f):
        pass

    def get_nn(self):
        pass

class P2PMap(PointWiseMap):
    """
    Point to point map, as an array or tensor of shape (n2,)
    """
    def __init__(self, p2p_21, n1=None):
        super().__init__()

        assert p2p_21.ndim == 1, "p2p should only have one dimension"
        self.p2p_21 = p2p_21  # (n2, )
        self.n2 = self.p2p_21.shape[0]
        self.n1 = n1

        self.max_ind = self.p2p.max() if n1 is None else n1-1
     
    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """
        if f.shape[0] <= self.max_ind:
            raise ValueError(f'Function f doesn\'t have enough entries, need at least {1+self.max_ind} but only has {f.shape[0]}')
        
        if f.ndim == 1 or f.ndim == 2:
            f_pb = f[self.p2p_21]  # (n2, k)
        elif f.ndim == 3:
            f_pb = f[:, self.p2p_21]  # (B, n2, k)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')
    
        return f_pb

    def get_nn(self):
        return self.p2p_21

class PreciseMap(PointWiseMap):
    """
    Point to barycentric map, using vertex to face and barycentric coordinates.
    """
    def __init__(self, v2face_21, bary_coords, faces1):
        super().__init__()

        # assert P21.ndim == 2, "Precise map should only have two dimension"

        self.v2face_21 = v2face_21
        self.bary_coords = bary_coords  # (N2, 3)
        self.faces1 = faces1

        self.n2 = self.v2face_21.shape[0]
        self.n1 = self.faces1.max()+1

        self._nn_map = None
     
    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """
       
        # f_pb = self.P21 @ f
        if f.ndim == 1 or f.ndim == 2:
            f_selected = f[self.faces1[self.v2face_21]]  # (N2, 3, p) or (N2, 3)
            # print('Selected', f_selected.max(), self.bary_coords.sum(1).max())
            if f.ndim == 1:
                f_pb = (self.bary_coords * f_selected).sum(1)
            else:
                f_pb = (self.bary_coords.unsqueeze(-1) * f_selected).sum(1)
                # print('Selected2', f_pb.max())

        elif f.ndim == 3:
            f_selected = f[: self.faces1[self.v2face_21]]  # (B, N2, 3, p)
            f_pb = (self.bary_coords.unsqueeze(0).unsqueeze(-1) * f_selected).sum(1)
    
        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = th.take_along_dim(self.faces1[self.v2face_21],
                                             self.bary_coords.argmax(1, keepdims=True),
                                             1).squeeze(-1)
            # self._nn_map = nn_query(self.emb1, self.emb2)
        
        return self._nn_map
        # return self.P21

class EmbP2PMap(P2PMap):
    """
    Point to point map, computed from embeddings.
    """
    def __init__(self, emb1, emb2):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)

        p2p_21 = nn_query(self.emb1, self.emb2)
        
        super().__init__(p2p_21, n1=self.emb1.shape[-2])


class EmbPreciseMap(PreciseMap):
    """
    Point to barycentric map, computed from embeddings.
    """
    def __init__(self, emb1, emb2, faces1, clear_cache=True):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)

        
        v2face_21, bary_coords = nn_query_precise(self.emb1, faces1, self.emb2, return_dist=False, batch_size=min(2000, emb2.shape[0]), clear_cache=clear_cache)

        # th.cuda.empty_cache()

        super().__init__(v2face_21, bary_coords, faces1)

class KernelDistMapOld(PointWiseMap):
    """
    Map of the the shape exp(- ||X_i - Y_j||_2^2 / blur**2)). Normalized per row.
    """
    def __init__(self, emb1, emb2, normalize=False, blur=None):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)
        

        self.blur = th.ones(1, device=self.emb1.device)
        
        if blur is not None:
            self.blur = self.blur * blur

        if normalize:
            with th.no_grad():
                self.blur = self.blur * th.sqrt(self.get_maxnorm())

        self.n1 = self.emb1.shape[-2]
        self.n2 = self.emb2.shape[-2]

        # self.batched1 = self.emb1.ndim == 3
        # self.batched2 = self.emb2.ndim == 3

        self.pull_back_formula = self.get_pull_back_formula()
        self._nn_map = None
    
    def get_maxnorm(self):
        formula = pykeops.torch.Genred('SqDist(X,Y)',
                    [f'X = Vi({self.emb1.shape[-1]})',          # First arg  is a parameter,    of dim 1
                    f'Y = Vj({self.emb2.shape[-1]})',          # Second arg is indexed by "i", of dim
                    ],
                    reduction_op='Max',
                    axis=0)

        max_dist = formula(self.emb1, self.emb2).max()

        return max_dist.squeeze()
    
    def get_pull_back_formula(self):
        """
        B, N1, 1 -> B, N2, 1
        """

        f = pykeops.torch.Vj(0, 1)  # (B, 1, N1, p)
        emb1_j = pykeops.torch.Vj(1, self.emb1.shape[1])  # (1, 1, N1, K)
        emb2_i = pykeops.torch.Vi(2, self.emb1.shape[1])  # (1, N2, 1, K)
        sqblur = pykeops.torch.Pm(3, 1)  # (B, 1)

        dist = -emb2_i.sqdist(emb1_j) / sqblur  # (B, N2, N1)

        return dist.sumsoftmaxweight(f, axis=1)


    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """
        
        n_func = f.shape[-1] if f.ndim > 1 else 1

        sqblur = 2*th.square(self.blur)

        # print(self.emb2.shape, self.emb1.shape, f.shape)
        # test = ((self.emb2.unsqueeze(1) * self.emb1.unsqueeze(0)).sum(-1).unsqueeze(-1) * f.unsqueeze(0)).sum(1)
        # print(test.shape, f.shape, self.emb2.shape)
        # return test

        if f.ndim == 1:
            f_pb = self.pull_back_formula(f.unsqueeze(-1), self.emb1, self.emb2, sqblur).squeeze(-1)  # (N2, )
        
        elif f.ndim == 2:
            f_input = f.transpose(0,1).contiguous()  # (p, N)
            f_pb = self.pull_back_formula(f_input, self.emb1.unsqueeze(0), self.emb2.unsqueeze(0), sqblur).squeeze(-1)  # (p, N2)
            f_pb = f_pb.transpose(0,1)  # (N2, p)
        
        elif f.ndim == 3:
            f_input = rearrange(f, 'B N p -> (B p) N').contiguous()
            f_pb = self.pull_back_formula(f_input, self.emb1.unsqueeze(0), self.emb2.unsqueeze(0), sqblur).squeeze(-1)  # (Bp, N2)
            f_pb = rearrange(f_pb, '(B p) N -> B N p', p=n_func)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')
        
        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = nn_query(self.emb1, self.emb2)
        
        return self._nn_map


class KernelDistMap(PointWiseMap):
    """
    Map of the the shape exp(- ||X_i - Y_j||_2^2 / blur**2)). Normalized per row.
    """
    def __init__(self, emb1, emb2, normalize=False, blur=None):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)
        

        self.blur = th.ones(1, device=self.emb1.device)
        #print(self.blur, blur)
        if blur is not None:
            self.blur = self.blur * blur

        if normalize:
            with th.no_grad():
                self.blur = self.blur * th.sqrt(self.get_maxnorm())

        self.n1 = self.emb1.shape[-2]
        self.n2 = self.emb2.shape[-2]

        # self.batched1 = self.emb1.ndim == 3
        # self.batched2 = self.emb2.ndim == 3

        # self.pull_back_formula = self.get_pull_back_formula()
        self._nn_map = None
    
    def get_maxnorm(self):
        formula = pykeops.torch.Genred('SqDist(X,Y)',
                    [f'X = Vi({self.emb1.shape[-1]})',          # First arg  is a parameter,    of dim 1
                    f'Y = Vj({self.emb2.shape[-1]})',          # Second arg is indexed by "i", of dim
                    ],
                    reduction_op='Max',
                    axis=0)

        max_dist = formula(self.emb1, self.emb2).max()

        return max_dist.squeeze()
    
    def get_pull_back_formula(self, dim):
        """
        B, N1, 1 -> B, N2, 1
        """

        f = pykeops.torch.Vj(0, dim)  # (B, 1, N1, p)
        emb1_j = pykeops.torch.Vj(1, self.emb1.shape[1])  # (1, 1, N1, K)
        emb2_i = pykeops.torch.Vi(2, self.emb1.shape[1])  # (1, N2, 1, K)
        sqblur = pykeops.torch.Pm(3, 1)  # (B, 1)

        dist = -emb2_i.sqdist(emb1_j) / sqblur  # (B, N2, N1)

        return dist.sumsoftmaxweight(f, axis=1)  # (B, N2, p)


    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """
        
        n_func = f.shape[-1] if f.ndim > 1 else 1
        pull_back_formula = self.get_pull_back_formula(n_func)

        sqblur = 2*th.square(self.blur)

        # print(self.emb2.shape, self.emb1.shape, f.shape)
        # test = ((self.emb2.unsqueeze(1) * self.emb1.unsqueeze(0)).sum(-1).unsqueeze(-1) * f.unsqueeze(0)).sum(1)
        # print(test.shape, f.shape, self.emb2.shape)
        # return test

        if f.ndim == 1:
            f_pb = pull_back_formula(f.unsqueeze(-1), self.emb1, self.emb2, sqblur).squeeze(-1)  # (N2, )
        
        elif f.ndim == 2:
            f_input = f.contiguous()  # (p, N)
            # print(f'f {f.is_contiguous()}, emb1 {self.emb1.is_contiguous()}, emb2 {self.emb2.is_contiguous()}')
            f_pb = pull_back_formula(f_input, self.emb1, self.emb2, sqblur) # (N2, p)
            
            # exit(-1)
            # f_pb = f_pb.transpose(0,1)  # (N2, p)
        
        elif f.ndim == 3:
            # f_input = rearrange(f, 'B N p -> (B p) N').contiguous()
            f_pb = pull_back_formula(f, self.emb1.unsqueeze(0), self.emb2.unsqueeze(0), sqblur) # (B, N2, p)
            # f_pb = rearrange(f_pb, '(B p) N -> B N p', p=n_func)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')
        
        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = nn_query(self.emb1, self.emb2)
        
        return self._nn_map

class OTMapOld(PointWiseMap):
    """
    Map of OT with distance ||X_i - Y_j||_2^2
    Computed from dual potentials and in log domain.
    """
    def __init__(self, X_i, Y_j, f_i, g_j, alpha_i=None, beta_j=None, blur=None):
        self.X = X_i.contiguous()  # (N1, K)
        self.Y = Y_j.contiguous()  # (N2, K)
        
        self.uniform_i = alpha_i is None
        self.uniform_j = beta_j is None
        
        self.alpha_i = alpha_i  # (N1, )
        self.beta_j = beta_j  # (N2, )
    
        self.blur = th.ones(1, device=self.X.device)
        if blur is not None:
            self.blur = self.blur * blur

        self.n1 = self.X.shape[-2]
        self.n2 = self.Y.shape[-2]

        self.f_i = f_i   # (N1, )
        self.g_j = g_j  # (N2, )

        assert self.f_i.ndim==1, "oops"
        assert self.g_j.ndim==1, "oops"
     
        self.pull_back_formula = self.get_pull_back_formula()
        self.nn_formula = self.get_nn_formula()
        self._nn_map = None
    
    def get_pull_back_formula(self):
        """
        B, N1, 1 -> B, N2, 1
        """
        f = pykeops.torch.Vi(0, 1)

        f_i = pykeops.torch.Vi(1, 1)
        g_j = pykeops.torch.Vj(2, 1)

        x_i = pykeops.torch.Vi(3, self.X.shape[-1])
        y_j = pykeops.torch.Vj(4, self.X.shape[-1])

        eps =  pykeops.torch.Pm(5, 1)

        dist = (f_i + g_j - x_i.sqdist(y_j)/2) / eps

        if self.uniform_i:
            return dist.sumsoftmaxweight(f, axis=0)

        log_alpha_i = pykeops.torch.LazyTensor(th.log(self.alpha_i).unsqueeze(-1), axis=0)

        return (dist + log_alpha_i).logsumexp(weight=f, axis=0)


    def sumaxis(self, ind):
        f_i = pykeops.torch.Vi(0, 1)
        g_j = pykeops.torch.Vj(1, 1)

        x_i = pykeops.torch.Vi(2, self.X.shape[-1])
        y_j = pykeops.torch.Vj(3, self.X.shape[-1])

        eps =  pykeops.torch.Pm(4, 1)

        dist = (f_i + g_j - x_i.sqdist(y_j)) / eps

        if ind==0:
            if self.uniform_i:
                formula = dist.logsumexp(axis=ind)
            else:
                alpha_i = pykeops.torch.LazyTensor(self.alpha_i.unsqueeze(-1), axis=0)

        formula = dist.logsumexp(axis=ind)

        return formula(self.f_i.unsqueeze(-1), self.g_j.unsqueeze(-1),
                       self.X, self.Y, 2*th.square(self.blur)).squeeze(-1).exp()
    
    def sumi(self):
        return self.sumaxis(0)

    def sumj(self):
        return self.sumaxis(1)

    def get_nn_formula(self):
        f_i = pykeops.torch.Vi(0, 1)
        g_j = pykeops.torch.Vj(1, 1)

        x_i = pykeops.torch.Vi(2, self.X.shape[-1])
        y_j = pykeops.torch.Vj(3, self.X.shape[-1])

        eps =  pykeops.torch.Pm(4, 1)

        dist = (f_i + g_j - x_i.sqdist(y_j)/2) / eps

        if self.uniform_i:
            return dist.argmax(0)

        alpha_i = pykeops.torch.LazyTensor(self.alpha_i.unsqueeze(-1), axis=0)

        return (dist.exp() * alpha_i).argmax(0)

    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """
        n_func = f.shape[-1] if f.ndim > 1 else 1

        sqblur = 2*th.square(self.blur)

        if f.ndim == 1:
            f_pb = self.pull_back_formula(f.unsqueeze(-1),  # (N2, 1)
                                          self.f_i.unsqueeze(-1), self.g_j.unsqueeze(-1),  # (N1, 1), (N2, 1)
                                          self.X, self.Y,  # (N1, K), (N2, K)
                                          sqblur  # (1,)
                                          ).squeeze(-1)  # (N2, )
        
        elif f.ndim == 2:
            f_input = f.transpose(0,1).contiguous()  # (p, N2)
            f_pb = self.pull_back_formula(f_input,  # (p, N2)
                                          self.f_i.unsqueeze(-1).unsqueeze(0), self.g_j.unsqueeze(-1).unsqueeze(0),  # (1, N1, 1), (1, N2, 1)
                                          self.X.unsqueeze(0), self.Y.unsqueeze(0),  # (1, N1, K), (1, N2, K)
                                          sqblur  # (1,)
                                          ).squeeze(-1)  # (p, N2)
            f_pb = f_pb.transpose(0,1)  # (N2, p)
        
        elif f.ndim == 3:
            f_input = rearrange(f, 'B N p -> (B p) N').contiguous()
            f_pb = self.pull_back_formula(f_input,  # (Bp, N2)
                                          self.f_i.unsqueeze(-1).unsqueeze(0), self.g_j.unsqueeze(-1).unsqueeze(0),  # (1, N1, 1), (1, N2, 1)
                                          self.X.unsqueeze(0), self.Y.unsqueeze(0),  # (1, N1, K), (1, N2, K)
                                          sqblur  # (1,)
                                          ).squeeze(-1)  # (Bp, N2)
            f_pb = rearrange(f_pb, '(B p) N -> B N p', p=n_func)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')
        
        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = self.nn_formula(self.f_i.unsqueeze(-1), self.g_j.unsqueeze(-1),
                                           self.X, self.Y, 2*th.square(self.blur)).squeeze(-1)
        
        return self._nn_map


class OTMap(PointWiseMap):
    """
    Map of OT with distance ||X_i - Y_j||_2^2
    Computed from dual potentials and in log domain. Accept weights.
    """
    def __init__(self, X_i, Y_j, f_i, g_j, alpha_i=None, beta_j=None, blur=None):
        self.X = X_i.contiguous()  # (N1, K)
        self.Y = Y_j.contiguous()  # (N2, K)
        
        self.uniform_i = alpha_i is None
        self.uniform_j = beta_j is None
        
        self.alpha_i = alpha_i  # (N1, )
        self.beta_j = beta_j  # (N2, )
    
        self.blur = th.ones(1, device=self.X.device)
        if blur is not None:
            self.blur = self.blur * blur

        self.n1 = self.X.shape[-2]
        self.n2 = self.Y.shape[-2]

        self.f_i = f_i   # (N1, )
        self.g_j = g_j  # (N2, )

        assert self.f_i.ndim==1, "oops"
        assert self.g_j.ndim==1, "oops"
     
        self.pull_back_formula = self.get_pull_back_formula()
        self.nn_formula = self.get_nn_formula()
        self._nn_map = None
    
    def get_pull_back_formula(self):
        """
        B, N1, 1 -> B, N2, 1
        """
        f = pykeops.torch.Vi(0, 1)  # (p, N1, 1, 1)

        f_i = pykeops.torch.Vi(1, 1)  # (1, N1, 1, 1)
        g_j = pykeops.torch.Vj(2, 1)  # (1, 1, N2, 1)

        x_i = pykeops.torch.Vi(3, self.X.shape[-1])  # (1, N1, 1, K)
        y_j = pykeops.torch.Vj(4, self.X.shape[-1])  # (1, 1, N2, K)

        eps =  pykeops.torch.Pm(5, 1)

        dist = (f_i + g_j - x_i.sqdist(y_j)/2) / eps  # (1, N1, N2)

        if self.uniform_i:
            return dist.sumsoftmaxweight(f, axis=0)

        log_alpha_i = pykeops.torch.LazyTensor(th.log(self.alpha_i).unsqueeze(0).unsqueeze(-1).unsqueeze(-1))  # (1, N1, 1, 1)
        # alpha_i = pykeops.torch.LazyTensor(self.alpha_i.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))  # (1, N1, 1, 1)

        return (dist + log_alpha_i).sumsoftmaxweight(f, axis=1)
        # return dist.logsumexp(weight=f*alpha_i, axis=0)


    def sumaxis(self, ind):
        f_i = pykeops.torch.Vi(0, 1)
        g_j = pykeops.torch.Vj(1, 1)

        x_i = pykeops.torch.Vi(2, self.X.shape[-1])
        y_j = pykeops.torch.Vj(3, self.X.shape[-1])

        eps =  pykeops.torch.Pm(4, 1)

        dist = (f_i + g_j - x_i.sqdist(y_j)/2) / eps

        if ind==0:
            if not self.uniform_i:
                log_alpha_i = pykeops.torch.LazyTensor(th.log(self.alpha_i).unsqueeze(-1), axis=0)  # (N1, 1, 1)
                dist = dist + log_alpha_i
            

            formula = dist.logsumexp(axis=ind)

            res = formula(self.f_i.unsqueeze(-1), self.g_j.unsqueeze(-1),
                             self.X, self.Y, th.square(self.blur)).squeeze(-1).exp()
            
            if self.uniform_i:
                res = res / self.n1

            return res

        # formula = dist.logsumexp(axis=ind)

        return 
    
    def sumi(self):
        return self.sumaxis(0)

    def sumj(self):
        return self.sumaxis(1)

    def get_nn_formula(self):
        f_i = pykeops.torch.Vi(0, 1)  # (N1, 1, 1)
        g_j = pykeops.torch.Vj(1, 1)  # (1, N2, 1)

        x_i = pykeops.torch.Vi(2, self.X.shape[-1])  # (N1, 1, K)
        y_j = pykeops.torch.Vj(3, self.X.shape[-1])  # (1, N2, K)

        eps =  pykeops.torch.Pm(4, 1)

        dist = (f_i + g_j - x_i.sqdist(y_j)/2) / eps

        if self.uniform_i:
            return dist.argmax(0)

        # alpha_i = pykeops.torch.LazyTensor(self.alpha_i.unsqueeze(-1), axis=0)    # (N1, 1, 1)
        log_alpha_i = pykeops.torch.LazyTensor(th.log(self.alpha_i).unsqueeze(-1), axis=0)  # (N1, 1, 1)

        return (dist + log_alpha_i).argmax(0)

    def pull_back(self, f):
        """
        Pull back f using the map T.

        Parameters:
        ------------------
        f : (N1,), (N1, p) or (B, N, p)

        Output
        -------------------
        pull_back : (N2, p)  or (B, N2, p)
        """
        n_func = f.shape[-1] if f.ndim > 1 else 1

        sqblur = th.square(self.blur)

        if f.ndim == 1:
            f_pb = self.pull_back_formula(f.unsqueeze(-1).unsqueeze(0),  # (1, N2, 1)
                                          self.f_i.unsqueeze(-1).unsqueeze(0), self.g_j.unsqueeze(-1).unsqueeze(0),  # (1, N1, 1), (1, N2, 1)
                                          self.X.unsqueeze(0), self.Y.unsqueeze(0),  # (1, N1, K), (1, N2, K)
                                          sqblur  # (1,)
                                          ).squeeze(-1).squeeze(0)  # (N2, )
        
        elif f.ndim == 2:
            f_input = f.transpose(0,1).contiguous()  # (p, N2)
            f_pb = self.pull_back_formula(f_input,  # (p, N2)
                                          self.f_i.unsqueeze(-1).unsqueeze(0), self.g_j.unsqueeze(-1).unsqueeze(0),  # (1, N1, 1), (1, N2, 1)
                                          self.X.unsqueeze(0), self.Y.unsqueeze(0),  # (1, N1, K), (1, N2, K)
                                          sqblur  # (1,)
                                          ).squeeze(-1)  # (p, N2)
            f_pb = f_pb.transpose(0,1)  # (N2, p)
        
        elif f.ndim == 3:
            f_input = rearrange(f, 'B N p -> (B p) N').contiguous()
            f_pb = self.pull_back_formula(f_input,  # (Bp, N2)
                                          self.f_i.unsqueeze(-1).unsqueeze(0), self.g_j.unsqueeze(-1).unsqueeze(0),  # (1, N1, 1), (1, N2, 1)
                                          self.X.unsqueeze(0), self.Y.unsqueeze(0),  # (1, N1, K), (1, N2, K)
                                          sqblur  # (1,)
                                          ).squeeze(-1)  # (Bp, N2)
            f_pb = rearrange(f_pb, '(B p) N -> B N p', p=n_func)
        else:
            raise ValueError('Function is only dim 1, 2 or 3')
        
        # if not self.uniform_i:
        #     f_pb = f_pb.exp()
        
        return f_pb

    def get_nn(self):
        if self._nn_map is None:
            self._nn_map = self.nn_formula(self.f_i.unsqueeze(-1), self.g_j.unsqueeze(-1),
                                           self.X, self.Y, th.square(self.blur)).squeeze(-1)
        
        return self._nn_map


class EmbOTMap(OTMap):
    """
    Map of OT with distance ||X_i - Y_j||_2^2
    Compute from embeddings.
    """
    def __init__(self, emb1, emb2, weights1=None, weights2=None, normalize=False, blur=None, compute_diam=False, scaling=.5, nit=None):
        self.emb1 = emb1.contiguous()  # (N1, K)
        self.emb2 = emb2.contiguous()  # (N2, K)
        
        self.use_weights = weights1 is not None
        if weights1 is not None:
            with th.no_grad():
                alpha_i = weights1 / weights1.sum()
                beta_j = weights2 / weights2.sum()
        else:
            alpha_i = None
            beta_j = None

        blur = 1 if blur is None else blur

        
        diameter, max_norm = self._get_diameter_and_maxnorm(normalize, compute_diam)
        
        otloss = SamplesLoss(loss="sinkhorn", blur=blur, debias=False, potentials=True, backend="online", diameter=diameter, scaling=scaling)
        
        input1 = self.emb1 / max_norm if normalize else self.emb1
        input2 = self.emb2 / max_norm if normalize else self.emb2
        if self.use_weights:
            f_i, g_j = otloss(alpha_i, input1, beta_j, input2, nit=nit)
        else:
            f_i, g_j = otloss(input1, input2, nit=nit)
        
        # print(f_i[0].shape, g_j[0].shape, f_i[0].ndim, g_j[0].ndim)

        super().__init__(input1, input2, f_i[0], g_j[0], alpha_i=alpha_i, beta_j=beta_j, blur=blur)
    
    def _get_diameter_and_maxnorm(self, normalize, compute_diam):
        if normalize or compute_diam:
            with th.no_grad():
                max_norm = th.sqrt(self.get_maxsqnorm())
        else:
            max_norm = None

        if not normalize and compute_diam:
            diameter = max_norm.cpu().item()
        elif normalize:
            diameter = 1
        else:
            diameter = None

        return diameter, max_norm

    def get_maxsqnorm(self):
        formula = pykeops.torch.Genred('SqDist(X,Y)',
                    [f'X = Vi({self.emb1.shape[-1]})',          # First arg  is a parameter,    of dim 1
                    f'Y = Vj({self.emb2.shape[-1]})',          # Second arg is indexed by "i", of dim
                    ],
                    reduction_op='Max',
                    axis=0)

        max_dist = formula(self.emb1, self.emb2).max()

        return max_dist.squeeze()