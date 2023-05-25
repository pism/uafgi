##import pytest
import numpy as np
import scipy
from osgeo import osr
from uafgi.util import gisutil#,sparseutil
from uafgi.elevclass import overlap_l0,ecmaker,ecvanilla
#import pyproj

def mequals(mat, data, row, col):
    """Test if a matrix equals a specific value"""

    # Sort indices the same in both matrices
    mat = mat.tocsr().tocoo()
    mat2 = scipy.sparse.coo_matrix((data, (row, col))).tocsr().tocoo()

    assert np.all(mat.data == mat2.data)
    assert np.all(mat.row == mat2.row)
    assert np.all(mat.col == mat2.col)


def test_elevclass():

    # Set up the system
    dxI = 1.0
    dyI = 1.0
    dxA = 2*dxI
    dyA = 2*dyI
    # gridA is 3 tall; the top-most row does not intersect with I
    gridA = gisutil.RasterInfo(gisutil.ortho_wkt, 1, 2, [0,dxA,0,  2*dyA,0,dyA])
    gridI = gisutil.RasterInfo(gisutil.ortho_wkt, 2, 4, [0,dxI,0,  4*dyI,0,dyI])

    mask_inI = np.ones((gridI.ny, gridI.nx), dtype=bool)
    mask_inI[1,1] = False    # One I grid cell is not used
    #mask_inI[:] = True

    # Obtain an overlap matrix based on nearest neighbor
    # constant-value polygon gridcells in Carteisan space.
    IuA = overlap_l0.nearest_neighbors(gridI, mask_inI, gridA)
    print('------------- IuA')
    print(IuA)
    mequals(IuA, [1.,1.,1.,1.,1.,1.,1.], [0,1,2,4,5,6,7], [0,0,0,1,1,1,1])

    # -----------------------------------------------
    # Make up a DEM
    elevI = np.arange(gridI.nxy, dtype='d')
    hcdefs = np.array([0., 1.5, 5.5, 8.0])
    ms = ecvanilla.MatrixSet(IuA, elevI, gridA, hcdefs)

    # EvA should be all 1's
    assert np.all(np.abs(ms.EvA.data - 1.0) < 1e-13)

    # Sum over each EC should sum up to 1
    # (Or if the EC is unused, sum to 0)
    wEvI = ecmaker.wIuJ(ms.EvI)
    unused_ecs = [3,4]
    assert np.all(wEvI[unused_ecs] == 0)
    mask = np.ones(wEvI.size, dtype=bool)
    mask[unused_ecs] = False
    assert np.all(np.abs(wEvI[mask] - 1.0) < 1e-10)


def test_elevclass2():
    """Test where the I grid goes beyond the A grid"""
    # Set up the system
    dxI = 1.0
    dyI = 1.0
    dxA = 2*dxI
    dyA = 2*dyI
    # gridA is 3 tall; the top-most row does not intersect with I
    gridA = gisutil.RasterInfo(gisutil.ortho_wkt, 1, 1, [0,dxA,0,  2*dyA,0,dyA])
    gridI = gisutil.RasterInfo(gisutil.ortho_wkt, 2, 4, [0,dxI,0,  4*dyI,0,dyI])

    mask_inI = np.ones((gridI.ny, gridI.nx), dtype=bool)
    mask_inI[1,1] = False    # One I grid cell is not used
    #mask_inI[:] = True

    # Obtain an overlap matrix based on nearest neighbor
    # constant-value polygon gridcells in Carteisan space.
    IuA = overlap_l0.nearest_neighbors(gridI, mask_inI, gridA)
    print('------------- IuA')
    print(IuA)


    # Make up a DEM
    elevI = np.arange(gridI.nxy, dtype='d')
    hcdefs = np.array([0., 1.5, 5.5, 8.0])
    ms = ecvanilla.MatrixSet(IuA, elevI, gridA, hcdefs)

    # TODO: Test that the matrices in test_elevclass2() are subsets of
    # matrices in test_elevclass()

test_elevclass()
