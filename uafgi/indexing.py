# Copied from Python at:
#     https://github.com/citibeth/icebin/blob/0a05e9421f61a277c963c785730bdc06727ca025/pylib/icebin/ibgrid.py
# Original C++ code at:
#    https://github.com/citibeth/ibmisc/blob/710c52b1ea92e73a5ce2ff3116989d8162b2f161/slib/ibmisc/indexing.hpp

import numpy as np
import functools
import operator

# -------------------------------------------------------
class Indexing(object):
    def __init__(self, base, extent, indices):
        self.base = np.array(base)
        self.extent = np.array(extent)
        self.indices = np.array(indices)

        # Turn it all into Numpy arrays
        if not isinstance(self.base, np.ndarray):
            self.base = np.array([self.base], dtype=type(self.base))
        if not isinstance(self.extent, np.ndarray):
            self.extent = np.array([self.extent], dtype=type(self.extent))
        if not isinstance(self.indices, np.ndarray):
            self.indices = np.array([self.indices], dtype=type(self.indices))

        self.size = 1
        for ex in self.extent:
            self.size *= ex

        # Shape of a row-major array in memory
        # Row-order ordered indices
        if (self.indices[0] == 0):
            self.shape = self.extent
        else:
            self.shape = self.extent[::-1]    # Reverse

        self.make_strides()

    def __len__(self):
        return functools.reduce(operator.mul, self.extent)


    @property
    def rank(self):
        return len(self.extent)

    def make_strides(self):
        rank = self.rank
        self.strides = np.zeros((rank,),dtype='i')
        self.strides[self.indices[rank-1]] = 1;
        for d in range(rank-2,-1,-1):
            self.strides[self.indices[d]] = self.strides[self.indices[d+1]] * self.extent[self.indices[d+1]];

    def tuple_to_index(self, tuple):
        ix = 0
        for k in range(0,self.rank):
            ix += (tuple[k]-self.base[k]) * self.strides[k];
        return ix;

    def index_to_tuple(self, ix):
        ix0 = ix
        tpl = np.zeros(len(self.shape),dtype='i')
        for d in range(0,self.rank-1):       # indices by descending stride
            k = self.indices[d]
            tpl_k = ix // self.strides[k]
            ix -= tpl_k*self.strides[k]
            tpl[k] = tpl_k + self.base[k]

        tpl[self.indices[self.rank-1]] = ix
        return tuple(tpl)

    def transpose(self):
        return Indexing(np.flipud(self.base), np.flipud(self.extent), np.flipud(self.indices))

