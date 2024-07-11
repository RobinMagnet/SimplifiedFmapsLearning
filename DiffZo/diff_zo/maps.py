import sys
import os
from pathlib import Path

import torch as th
import pykeops

from einops import rearrange


SOURCE_DIR = Path(os.path.abspath(__file__)).parents[1]
sys.path.append(os.path.join(SOURCE_DIR, 'ProjectionUtils'))
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

