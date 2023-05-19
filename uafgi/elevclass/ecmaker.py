import numpy as np
import scipy.sparse

"""General utilities to create regridding matrices with elevation classes"""

def extend_to_elev(IuA, elevI, hcdefs):
    """Builds an interpolation matrix between grid I and the elevation grid E.
    gridI:
        Could be the basic fine-scale grid (ice grid), or the exchange
        grid if exact polygon overlaps are being used.
    elevI: np.array(gridI.nxy)
        Hi-res elevations.  (NOt sure:::???/ Dense indexing, ALL gridcells should exist!)
    gridA:
        The coarse grid the elevation classes are based on
    hcdefs:
        Elevations at which we compute
    """
    # Consider using np.digitze here instead()???
    upper_ecI = np.searchsorted(hcdefs, elevI[IuA.ii])    # Upper elevation class for each gridcell IuA matrix
    lower_ecI = upper_ecI - 1

    intervalsI = hcdefs[upper_ecI] - hcdefs[lower_ecI] # Size EC interval each point is in
    lower_weightsI = (hcdefs[upper_ecI] - elevI) / intervalsI
    upper_weightsI = 1.0 - lower_weightsI

    # Index of each gridcell in E
    lower_eixI = IuA.jj * len(hcdefs) + lower_ecI
    lower_dataI = lower_weightsI * IuA.data

    upper_eixI = IuA.jj * len(hcdefs) + upper_ecI
    upper_dataI = upper_weightsI * IuA.data

    return coo_list(
        np.concatenate((lower_dataI, upper_dataI)),
        np.concatenate((IuA.ii, IuA.ii)),
        np.concatenate((lower_eixI, upper_eixI)))

# --------------------------------------------------------------------------
def diag(diagonal):
    """A simple sparse diagonal matrix creator"""
    return scipy.sparse.diags([diagonal], [0])
_diag = diag    # Internal name

def wIuJ(IuJ):
    """Compute sum of rows"""
    return np.squeeze(np.asarray(IuJ.sum(axis=1)))

def IuJw(IuJ):
    """Compute sum of columns"""
    return np.squeeze(np.asarray(IuJ.sum(axis=2)))

def sIuJ(IuJ, diag=True):
    """Compute sum of rows"""
    wI = wIuJ(IuJ)
    mask_out = (wI == 0)
    wI[mask_out] = 1    # Avoid RuntimeWarning: divide by zero
    sI = np.reciprocal(wI)
    sI[mask_out] = 0
    if diag:
        return _diag(sI)
    return sI

def IuJs(IuJ, diag=True):
    """Compute sum of rows"""
    wJ = IuJw(IuJ)
    mask_out = (wJ == 0)
    wJ[mask_out] = 1    # Avoid RuntimeWarning: divide by zero
    sJ = np.reciprocal(wJ)
    sJ[mask_out] = 0
    if diag:
        return _diag(sJ)
    return sJ




def scale(IuJ):
    """Returns: IvJ
    IuJ_list: coo_list
        Unscaled matrix, raw output of matrix generators
    IvJ: scipy.coo_matrix
        Scaled matrix, ready to use
    Returns:
        If transpose:
            IvJ
        Elase:
            JvI
    """
    # Or try this: https://stackoverflow.com/questions/52953231/numpy-aggregate-into-bins-then-calculate-sum
    return sIuJ(IuJ, diag=True) * IuJ
