import os.path as osp
import sys
import numpy as np
from pathlib import Path

from .faust import ShapeDataset as FaustShapeDataset
from .faust import ShapePairDataset


class ShapeDataset(FaustShapeDataset):
    TRAIN_IDX = np.arange(0, 51)
    TEST_IDX = np.arange(51, 71)
