import numpy as np
import netCDF4
import os,subprocess
from cdo import Cdo
from uafgi import nsidc,cgutil,gdalutil,shapelyutil
from uafgi import giutil,cdoutil,make,ioutil,gicollections
import pandas as pd
import skimage.segmentation

# Fjord pixel classifcation types
UNUSED = 0
LOWER_FJORD = 1
TERMINUS = 2
TERMINUS_EXTRA = 3    # Terminus line, outside of fjord
UPPER_FJORD = 4
UPPER_FJORD_SEED = 5

# Translate fjord classifications into lower / upper fjord;
# either including or not including the terminus line.
# LT = Less than (as in FORTRAN)
# LE = Less than or equal
# GE = Greater than or equal
# GT = Greater than
# Use:
#    fjc = classify_fjord(...)
#    lower_fjord = np.isin(fjc, glacier.LT_TERMINUS)
#    upper_fjord = np.isn(fjc, glacier.GE_TERMINUS)
LT_TERMINUS = (LOWER_FJORD,)
LE_TERMINUS = (LOWER_FJORD, TERMINUS)
GE_TERMINUS = (TERMINUS, UPPER_FJORD, UPPER_FJORD_SEED)
GT_TERMINUS = (UPPER_FJORD, UPPER_FJORD_SEED)
ALL_FJORD = (LOWER_FJORD, TERMINUS, UPPER_FJORD, UPPER_FJORD_SEED)

def classify_fjord(fjord, grid_info, upstream_loc, terminus):
    """Splits a fjord along a terminus, into an upper and lower section.
    The upper portion does NOT include the (rasterized) terminus line

    fjord: np.array(bool)
        Definition of a fjord on a local grid.
        Eg: result of bedmachine.get_fjord()

    grid_info: gdalutil.FileInfo
        Definition of the grid used for fjord
        Eg: gdalutil.FileInfo(grid_file)

    upstream_loc: shapely.geometry.Point
        A single point in the upstream portion of the fjord

    terminus: shapely.geometry.LineString
        The terminus on which to split

    Returns: np.array(int)
        0 = Unused
        1 = lower fjord
        2 = glacier terminus (in fjord)
        3 = glacier terminus (out of fjord)
        4 = upper fjord
        5 = the fill seed point (in the upper fjord)

    """

    # Extend and rasterize the terminus; can be used to cut fjord
    terminus_extended=cgutil.extend_linestring(terminus, 100000.)
    terminus_xr = gdalutil.rasterize_polygons(
        shapelyutil.to_datasource(terminus_extended), grid_info)

    # Cut the fjord with the terminus
    fj = np.zeros(fjord.shape)
    fj[fjord] = LOWER_FJORD
    fj[terminus_xr != 0] = TERMINUS_EXTRA
    fj[np.logical_and(terminus_xr != 0, fjord)] = TERMINUS

    # Position of upstream point on the raster
    seed = grid_info.to_ij(upstream_loc.x, upstream_loc.y)

    # Don't fill through diagonals
    selem = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])

    fj = skimage.segmentation.flood_fill(
        fj, (seed[1],seed[0]), UPPER_FJORD, selem=selem)
    fj[seed[1],seed[0]] = UPPER_FJORD_SEED
    return fj

