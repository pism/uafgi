import os,subprocess
import netCDF4
from uafgi import make,ncutil,gdalutil,shapelyutil,cdoutil
from uafgi.make import ncmake
import re
from osgeo import ogr
import numpy as np


def fixup_for_pism(ifname, ofname, tdir):
    """Fixup the global bedmachine file for use as PISM input file
    ifname:
        Name of BedMachine file
    odir:
        Place to put output BedMachine file
    tdir:
        Temporary directory
    """

    tmp1 = tdir.opath(ifname, '_fixup_pism1.nc')

    # Reverse the direction of the Y coordinate
    cmd = ['ncpdq', '-O', '-a', '-y', ifname, tmp1]
    subprocess.run(cmd, check=True)

    # Compress
    os.makedirs(os.path.split(ofname)[0], exist_ok=True)
    cdoutil.compress(tmp1, ofname)


def replace_thk(bedmachine_file0, bedmachine_file1, thk):
    """Copies bedmachine_file0 to bedmachine_file1, using thk in place of original 'thickness'
    bedmachien_file0:
        Name of original BedMachine file
    bedmachine_file1:
        Name of output BedMachine file
    thk:
        Replacement thickness field"""

    with netCDF4.Dataset(bedmachine_file0, 'r') as nc0:
        with netCDF4.Dataset(bedmachine_file1, 'w') as ncout:
            cnc = ncutil.copy_nc(nc0, ncout)
            vars = list(nc0.variables.keys())
            cnc.define_vars(vars)
            for var in vars:
                if var not in {'thickness'}:
                    cnc.copy_var(var)
            ncout.variables['thickness'][:] = thk


def get_fjord_gd(bmlocal_file, fj_poly):
    """Returns a raster of the fjord for a glacier.

    bmlocalfile:
        Localized Bedmachine file containing the glacier.
    fj_poly:
        Hand-drawn Shapely polygon containing the fjord.

    Returns: np.array(bool)
        Localized raster: True in fjord, False elsewhere
        Raster is in GDAL convention
    """


#    # Create an OGR in-memory datasource with our single polygon
#    fj_ds=ogr.GetDriverByName('MEMORY').CreateDataSource('memData')
#    fj_layer = fj_ds.CreateLayer('', None, ogr.wkbPolygon)
#    feat = ogr.Feature(fj_layer.GetLayerDefn())
#    feat.SetGeometry(ogr.CreateGeometryFromWkb(fj_poly.wkb))
 #   fj_layer.CreateFeature(feat)
    fj_ds = shapelyutil.to_datasource(fj_poly)

    # Rasterize the approx. trough polygon
    fb = gdalutil.FileInfo(bmlocal_file)
    approx_trough = gdalutil.rasterize_polygons(fj_ds, fb)

    # Read the bed from bedmachine
    with netCDF4.Dataset(bmlocal_file) as nc:
        bed = nc.variables['bed'][:]
        yy = nc.variables['y'][:]
    # Conform to GDAL conventions.
    # NOTE: It would be better to just use GDAL to read this array.
    if yy[1] - yy[0] > 0:
        bed = np.flipud(bed)


    # Intersect the appxorimate trough with the below-sea-level areas.
    # This gives the mask of
    # points where we check to see ice added/removed
    # during course of a run.
    fjord = (bed < 0)
    this_fjord = (np.logical_and(fjord, approx_trough != 0)).data

    return this_fjord

