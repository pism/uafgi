import os
import numpy as np
import scipy.sparse
from akramms import config,params
from uafgi.util import gdalutil,wrfutil
from osgeo import gdalconst

def nearest_neighbors(gridI, mask_inI, gridA):
    """Computes the overlap matrix IoA.  Assumes gridcells in I are
    really small (compared to A), and assigns each 100% to the nearest
    gridcell in A, even for gridcells in I on the border between two
    gridcells in A.  I may be incomplete, it is assumed A covers the
    entire domain of I.

    gridI, gridA:
        Grid definitions for I and A grid
    mask_inI: bool(ny,nx)
        True if gridell is included in the grid, False if not.
        Eg: np.ones((gridI.ny, gridI.nx), dtype=bool)
    Returns: IuA
    """
    # Assume 1D indexing for all arrays, but allow user to pass in 2D
    mask_inI = mask_inI.reshape(-1)

    # aidA = Indices of A gridcells, on the A grid
    aidA = np.arange(gridA.nxy, dtype='i')    # "Data" to regrid contains unique ID of each gridcell.


    # Compute aidI --- indices of A gridcells, on the I grid
    aidA_2 = np.reshape(aidA, (gridA.ny, gridA.nx))
    print('aidA_2', aidA_2)
    aidI_2 = gdalutil.regrid(
        aidA_2, gridA, -1,    # A has data everywhere
        gridI, -1,    # nodata value won't be used
        resample_algo=gdalconst.GRA_NearestNeighbour).astype('i')
    aidI = aidI_2.reshape(-1)
    print('aidI_2 ', aidI_2)


    # iidI = Indices of I gridcells, on the I grid
    iidI = np.arange(gridI.nx*gridI.ny, dtype='i')

    iidI_2 = np.reshape(iidI, (gridI.ny, gridI.nx))
    print('iidI_2 ', iidI_2)
    print('mask_inI_2 ', mask_inI.reshape((gridI.ny, gridI.nx)))

    # --------- Masking: Remove useless gridcells from I
    # Remove gridcells in gridI that don't overlap anything in gridA
    xmask_inI = (aidI != -1)

    # Also remove gridcells in I that don't exist
    if mask_inI is not None:
        xmask_inI = np.logical_and(mask_inI, xmask_inI)

    # Apply the masking
    iidI = iidI[xmask_inI]
    aidI = aidI[xmask_inI]
    # ----------------------------------------------------

    # Compute area of gridcells --- in this case, all the same
    # TODO: Allow for variable-size grdcells, if needed
    areaI = np.zeros(iidI.shape) + (gridI.dx * gridI.dy)    # Area of gridcell

    print('areaI ', areaI)
    print('iidI ', iidI)
    print('aidI ', aidI)

    return scipy.sparse.coo_matrix(
        (areaI, (iidI, aidI)),
        shape=(gridI.nxy, gridA.nxy))

