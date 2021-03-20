import numpy as np
import netCDF4
import os,subprocess
from cdo import Cdo
from uafgi import nsidc,cgutil,gdalutil,shapelyutil
from uafgi import giutil,cdoutil,make,ioutil,gicollections
import pandas as pd
import skimage.segmentation

def upstream_fjord(fjord, grid_info, upstream_loc, terminus):
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
        2 = glacier terminus
        4 = upper fjord
        5 = the fill seed point (in the upper fjord)
    
    """

    # Extend and rasterize the terminus; can be used to cut fjord
    terminus_extended=cgutil.extend_linestring(terminus, 100000.)
    terminus_xr = gdalutil.rasterize_polygons(
        shapelyutil.to_datasource(terminus_extended), grid_info)

    # Cut the fjord with the terminus
    fj = np.zeros(fjord.shape)
    fj[fjord] = 1
    fj[terminus_xr != 0] = 2

    # Position of upstream point on the raster
    seed = grid_info.to_ij(upstream_loc.x, upstream_loc.y)

    # Don't fill through diagonals
    selem = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])

    fj = skimage.segmentation.flood_fill(fj, (seed[1],seed[0]), 4, selem=selem)
    fj[seed[1],seed[0]] = 5
    return fj

