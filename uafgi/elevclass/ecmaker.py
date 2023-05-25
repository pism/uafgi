import numpy as np
import scipy.sparse

"""General utilities to create regridding matrices with elevation classes"""

def IuE(IuA, elevI, hcdefs):
    """Builds an interpolation matrix between grid I and the elevation grid E.
    gridI:
        Could be the basic fine-scale grid (ice grid), or the exchange
        grid if exact polygon overlaps are being used.
    elevI: np.array(gridI.nxy)
        Hi-res elevations.  (NOt sure:::???/ Dense indexing, ALL gridcells should exist!)
    gridA:
        The coarse grid the elevation classes are based on
    hcdefs:
        Elevations at which we compute.  The following must hold:
        hcdefs[0] <= np.min(elevI)
        hcdefs[-1] > np.max(elevI)
    """
    elevI = elevI.reshape(-1)

    # Check all elevations are within range
    assert hcdefs[0] <= np.min(elevI)
    assert np.max(elevI) < hcdefs[-1]

    # Consider using np.digitze here instead()???
    elvI = elevI[IuA.row]    # Elevation of the gridcells from our matrix
    upper_ecI = np.searchsorted(hcdefs, elvI, side='right')    # Upper elevation class for each gridcell IuA matrix
    lower_ecI = upper_ecI - 1

    print('hcdefs ', hcdefs)
    print('elvI ', elvI)
    print('lower_ecI ', lower_ecI)
    print('upper_ecI ', upper_ecI)

    intervalsI = hcdefs[upper_ecI] - hcdefs[lower_ecI] # Size EC interval each point is in
    lower_weightsI = (hcdefs[upper_ecI] - elvI) / intervalsI
    upper_weightsI = 1.0 - lower_weightsI

    print('intervalsI ', intervalsI)
    print('lower_weightsI ', lower_weightsI)
    print('upper_weightsI ', upper_weightsI)

    # Index of each gridcell in E
    lower_eixI = IuA.col * len(hcdefs) + lower_ecI
    lower_dataI = lower_weightsI * IuA.data

    upper_eixI = IuA.col * len(hcdefs) + upper_ecI
    upper_dataI = upper_weightsI * IuA.data

    ret = scipy.sparse.coo_matrix(
        (np.concatenate((lower_dataI, upper_dataI)),
            (np.concatenate((IuA.row, IuA.row)),
            np.concatenate((lower_eixI, upper_eixI)))),
        shape=(IuA.shape[0], len(hcdefs)*IuA.shape[1]))
    ret.sum_duplicates()
    ret.eliminate_zeros()
    return ret

# --------------------------------------------------------------------------
def diag(diagonal):
    """A simple sparse diagonal matrix creator"""
    return scipy.sparse.diags([diagonal], [0])
_diag = diag    # Internal name

def wIuJ(IuJ):
    """Compute sum of rows"""
    ret = np.asarray(IuJ.sum(axis=1))
    return ret[:,0]
#    return np.squeeze(np.asarray(IuJ.sum(axis=1)))

def IuJw(IuJ):
    """Compute sum of columns"""
    ret = np.asarray(IuJ.sum(axis=2))
    return ret[0,:]
#    return np.squeeze(np.asarray(IuJ.sum(axis=2)))

def sIuJ(IuJ, diag=True):
    """Compute sum of rows"""
    wI = wIuJ(IuJ)
    mask_out = (wI == 0)
    wI[mask_out] = 1    # Avoid RuntimeWarning: divide by zero
    sI = np.reciprocal(wI)
    print('xxx ', sI)
    print('yyy ', mask_out)
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
