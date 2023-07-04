import pyproj
import numpy as np
import netCDF4
from uafgi.util import gisutil
from osgeo import gdal
import gridfill

# From the gridfill docs...
#def gridfill.fill(grids, xdim, ydim, eps, relax=.6, itermax=100, initzonal=False,
#         cyclic=False, verbose=False):
#    """
#    Fill missing values in grids with values derived by solving
#    Poisson's equation using a relaxation scheme.
#    **Arguments:**
#    *grid*
#        A masked array with missing values to fill.
#    *xdim*, *ydim*
#        The numbers of the dimensions in *grid* that represent the
#        x-coordinate and y-coordinate respectively.
#    *eps*
#        Tolerance for determining the solution complete.
#    **Keyword arguments:**
#    *relax*
#        Relaxation constant. Usually 0.45 <= *relax* <= 0.6. Defaults to
#        0.6.
#    *itermax*
#        Maximum number of iterations of the relaxation scheme. Defaults
#        to 100 iterations.
#    *initzonal*
#        If *False* missing values will be initialized to zero, if *True*
#        missing values will be initialized to the zonal mean. Defaults
#        to *False*.
#    *cyclic*
#        Set to *False* if the x-coordinate of the grid is not cyclic,
#        set to *True* if it is cyclic. Defaults to *False*.
#    *verbose*
#        If *True* information about algorithm performance will be
#        printed to stdout, if *False* nothing is printed. Defaults to
#        *False*.
#    """


# CRS used by WRF
grs1980_wkt = epsg4019_wkt = \
"""GEOGCS["Unknown datum based upon the GRS 1980 ellipsoid",
    DATUM["Not_specified_based_on_GRS_1980_ellipsoid",
        SPHEROID["GRS 1980",6378137,298.257222101,
            AUTHORITY["EPSG","7019"]],
        AUTHORITY["EPSG","6019"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4019"]]"""


def wrf_info(geo_fname):
    """
    geo_fname:
        Name of the WRF geometry definition file (eg: geosoutheast.nc)
    Returns: uafgi.util.RasterInfo
        Definition of WRF coordinate system / etc.
    """
    with netCDF4.Dataset(geo_fname) as nc:
        # Get lon/lat of center of each gridcell
        lon_m = nc.variables['XLONG_M'][0,:,:]
        lat_m = nc.variables['XLAT_M'][0,:,:]
        nji = lon_m.shape

        # NOTE: WRF NetCDF file is in north-down format; i.e. the most
        #       northernly points are at the END of the Y axis.  This
        #       is the opposite of typical GeoTIFF, and we would
        #       expect dy to be positive.

        # Obtain CRS for this WRF run
        MAP_PROJ = nc.MAP_PROJ
        if MAP_PROJ != 1:
            raise ValueError('WRF MAP_PROJ={} must be 1 (Lambert Concical LCC projection in PROJ.4).  Other values of MAP_PROJ are not supported'.format(MAP_PROJ))
        wrf_crs = pyproj.CRS.from_string(f'+proj=lcc +lat_1={nc.TRUELAT1} +lat_2={nc.TRUELAT2} +lat_0={nc.CEN_LAT} +lon_0={nc.CEN_LON} +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs')


    # Convert Gridcell centers from lon/lat to WRF's CRS
    lonlat_crs = pyproj.CRS.from_string(grs1980_wkt)
    ll2wrf = pyproj.Transformer.from_crs(lonlat_crs, wrf_crs, always_xy=True)
    xx_m_wrf, yy_m_wrf = ll2wrf.transform(lon_m,lat_m)

    # Set up the geotransform for the WRF raster
    # See: https://gdal.org/tutorials/geotransforms_tut.html
    dx = np.mean(xx_m_wrf[:,1:] - xx_m_wrf[:,0:-1])
    dy = np.mean(yy_m_wrf[1:,:] - yy_m_wrf[0:-1,:])
    gt_wrf = [
        np.mean(xx_m_wrf[:,0]) - .5*dx,    # edge of x-coord of origin (most westerly pixel)
        dx,                       # W-E pixel width
        0,                        # Row rotation
        # Projected coordinate space is north-up; whereas raster space (for WRF) is north-down.
        # Because those two are different, the 0.5 needs to have positive sign.
        # (If the yare the same, then we would use -0.5*dy)
        np.mean(yy_m_wrf[0,:]) + .5*dy,    # edge of y-coord of origin (most southerly pixel in this case)
        0,                        # Column rotation

        dy]                       # N-S pixel resolution (negative val for north-up images)

    return gisutil.RasterInfo(wrf_crs.to_wkt(), nji[1], nji[0], gt_wrf)

def read_raw(data_fname, vname, units=None, fill_holes=False):
    """Reads a WRF file with the corect geometry, etc.
    units: str (OPTIONAL)
        Convert to these units"""

    with netCDF4.Dataset(data_fname) as nc:
        # Masked array
        ncv = nc.variables[vname]
        orig_units = ncv.units
        #print('read_raw ', data_fname, vname)
        #print(ncv.__dict__)
        nodata_value = ncv._FillValue if hasattr(ncv, '_FillValue') else None
        masked_data = ncv[:,:]    # sx3(j=south_north,i=west_east)

    if fill_holes:
        data_rawunits, converged = gridfill.fill(masked_data, 1, 0, .1)#, itermax=10000)
    else:
        data_rawunits = np.ma.getdata(masked_data)
    if units is None:
        return data_rawunits,nodata_value
    return cfutil.convert(data_rawunits, orig_units, units),nodata_value

def read(data_fname, vname, geo_fname, units=None):
    """Read WRF dataset with same return as gdalutil.read_raster()
    data_fname:
        Name of raw WRF NetCDF file
    geo_fname:
        Name of the WRF geometry definition file (eg: geosoutheast.nc)
    Returns: grid_info, data
        grid_info: gisutil.RasterInfo
            Description of the raster file's grid
        data: np.array
            Data found in the raster file.
        nodata_value:
            Indicates no data for a gridcell in data
    """
    geo_info = wrf_info(geo_fname)
    data,nodata_value = read_raw(data_fname, vname, units=units)
    return geo_info, data, nodata_value


def write_geotiff(geo_info, data, ofname, flipud=True):
    """Writes a WRF raster to a GeoTIFF file.
    geo_info:
        Geometric info about the raster.
        Result of wrf_info() above
    data:
        The actual raster value.
        It must be with [j,i] indexing.
    ofname:
        File to write
    flipud:
        If True, flip the geometry and raster before saving.
        By default, WRF files are north-down.  With flipud=True
        by default, this generates GeoTIFF files is standard
        north-up format (although both work with QGIS).
        """

    (rows, cols) = data.shape
    gdalDriver = gdal.GetDriverByName('GTiff')
    outRaster = gdalDriver.Create(
        ofname, cols, rows, 1, gdal.GDT_Float32, ['COMPRESS=DEFLATE'])

    if flipud:
        data = np.flipud(data)
        geo_info = geo_info.flipud()

    try:
        outRaster.SetGeoTransform(geo_info.geotransform)
        outRaster.SetProjection(geo_info.wkt)
        outBand = outRaster.GetRasterBand(1)
        outBand.SetNoDataValue(np.nan)
        outBand.WriteArray(data)
    finally:
        outRaster = None

#class WRFTransformer:
#    def __init__(self, geo_fname, scene_wkt):
#
#        self.info = wrf_info(geo_fname)
#
#        # Obtain transfomer from scene coordinates to what WRF uses.
#        scene_crs = pyproj.CRS.from_string(scene_wkt)
#        wrf_crs = pyproj.CRS.from_string(self.info.wkt)
#        # There will be "error" in this because the spheroids do not match.
#        # WRF uses perfect sphere; whereas scene typically uses WGS84 or similar
#        self.scene2wrf = pyproj.Transformer.from_crs(scene_crs, wrf_crs, always_xy=True)
#
#    def to_ij(self, xx_scene, yy_scene):
#        """xx_scene, yy_scene:
#            Coordinates to convert, IN SCENE SPACE"""
#        xx_wrf,yy_wrf = self.scene2wrf.transform(xx_scene, yy_scene)
#        return self.info.to_ij(xx_wrf, yy_wrf)
#
#
