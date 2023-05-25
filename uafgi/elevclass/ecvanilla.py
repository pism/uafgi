import numpy as np
from functools import cached_property
from uafgi.elevclass import ecmaker

class MatrixSet:
    def __init__(self, IuA, elevI, gridA, hcdefs):
        self.IuA = IuA
        self.elevI = elevI
        self.gridA = gridA
        self.hcdefs = hcdefs

    @property
    def nhc(self):
        return len(self.hcdefs)

    def unpackE(self, ixE):
        """
        iE:
            Collection of 1D indices on the Elevation grid
        Returns: iA, ec
            iA: Each E gridcell on the A grid
            ec: Elevation class index of each E gridcell
        """
        return np.divmod(ixE, self.nhc)

    @cached_property
    def IuE(self):
        return ecmaker.IuE(self.IuA, self.elevI, self.hcdefs)

    @cached_property
    def IvA(self):
        return ecmaker.scale(self.IuA)

    @cached_property
    def AvI(self):
        return ecmaker.scale(self.IuA.transpose())

    @cached_property
    def IvE(self):
        return ecmaker.scale(self.IuE)

    @cached_property
    def EvI(self):
        return ecmaker.scale(self.IuE.transpose())

    @cached_property
    def EvA(self):
        return self.EvI * self.IvA

    @cached_property
    def AvE(self):
        return self.AvI * self.IvE
