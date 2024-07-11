import torch as th
import numpy as np
import scipy.sparse as sparse

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()

def toTH(array):
    """
    Convert a numpy array to a torch tensor
    """
    if 'float' in str(array.dtype):
        return th.from_numpy(array.astype(np.float32))
    
    return th.from_numpy(array)


# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return th.sparse_coo_tensor(th.LongTensor(indices), th.FloatTensor(values), th.Size(shape)).coalesce()


# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat

def sparse_th_from_args(*args):
    """
    Create a sparse tensor from arguments (indices, values, size).
    """
    return th.sparse_coo_tensor(*args).coalesce()