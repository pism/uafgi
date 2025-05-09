import pyproj
import numpy as np
import netCDF4
from uafgi.util import gisutil
from osgeo import gdal
import gridfill
import wrf    # wrf-python package

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

# Just use wrf-python...
#https://wrf-python.readthedocs.io/en/develop/user_api/generated/wrf.WrfProj.html#wrf.WrfProj.map_proj

#https://www.ncl.ucar.edu/Applications/wrflc.shtml
#https://www.ncl.ucar.edu/Applications/wrflc.shtml
#https://www2.mmm.ucar.edu/wrf/users/wrf_users_guide/build/html/wps.html#wps-namelist-variables
#https://proj.org/en/stable/operations/projections/ups.html
#MAP_PROJ = 0 --> "CylindricalEquidistant"
#MAP_PROJ = 1 --> "LambertConformal"
#MAP_PROJ = 2 --> "Stereographic"  (Polar Stereographic, or just 'polar')
#MAP_PROJ = 3 --> "Mercator"
#MAP_PROJ = 6 --> "Lat/Lon"


#// global attributes:
#                :TITLE = "OUTPUT FROM GEOGRID V4.5" ;
#                :SIMULATION_START_DATE = "0000-00-00_00:00:00" ;
#                :WEST-EAST_GRID_DIMENSION = 421 ;
#                :SOUTH-NORTH_GRID_DIMENSION = 451 ;
#                :BOTTOM-TOP_GRID_DIMENSION = 0 ;
#                :WEST-EAST_PATCH_START_UNSTAG = 1 ;
#                :WEST-EAST_PATCH_END_UNSTAG = 420 ;
#                :WEST-EAST_PATCH_START_STAG = 1 ;
#                :WEST-EAST_PATCH_END_STAG = 421 ;
#                :SOUTH-NORTH_PATCH_START_UNSTAG = 1 ;
#                :SOUTH-NORTH_PATCH_END_UNSTAG = 450 ;
#                :SOUTH-NORTH_PATCH_START_STAG = 1 ;
#                :SOUTH-NORTH_PATCH_END_STAG = 451 ;
#                :GRIDTYPE = "C" ;
#                :DX = 4000.f ;
#                :DY = 4000.f ;
#                :DYN_OPT = 2 ;
#                :CEN_LAT = 63.90041f ;
#                :CEN_LON = -152.2677f ;
#                :TRUELAT1 = 64.f ;
#                :TRUELAT2 = 1.e+20f ;
#                :MOAD_CEN_LAT = 63.99999f ;
#                :STAND_LON = -152.f ;
#                :POLE_LAT = 90.f ;
#                :POLE_LON = 0.f ;
#                :corner_lats = 55.13433f, 70.42133f, 70.51836f, 55.18422f, 55.13046f, 70.41382f, 70.51104f, 55.18046f, 55.11752f, 70.43813f, 70.53525f, 55.16738f, 55.11366f, 70.43061f, 70.52792f, 55.16362f ;
#                :corner_lons = -164.9492f, -176.0684f, -128.5916f, -139.4365f, -164.9786f, -176.1185f, -128.5409f, -139.407f, -164.9425f, -176.0908f, -128.5696f, -139.443f, -164.9718f, -176.141f, -128.519f, -139.4136f ;
#                :MAP_PROJ = 2 ;
#                :MMINLU = "MODIFIED_IGBP_MODIS_NOAH" ;
#                :NUM_LAND_CAT = 21 ;
#                :ISWATER = 17 ;
#                :ISLAKE = 21 ;
#                :ISICE = 15 ;
#                :ISURBAN = 13 ;
#                :ISOILWATER = 14 ;
#                :grid_id = 2 ;
#                :parent_id = 1 ;
#                :i_parent_start = 68 ;
#                :j_parent_start = 27 ;
#                :i_parent_end = 207 ;
#                :j_parent_end = 176 ;
#                :parent_grid_ratio = 3 ;
#                :sr_x = 1 ;
#                :sr_y = 1 ;
#                :FLAG_MF_XY = 1 ;
#                :FLAG_LAI12M = 1 ;
#                :FLAG_VAR_SSO = 1 ;
#                :FLAG_LAKE_DEPTH = 1 ;


        # Obtain CRS for this WRF run
        MAP_PROJ = nc.MAP_PROJ
        if MAP_PROJ == 1:
            wrf_crs = pyproj.CRS.from_string(f'+proj=lcc +lat_1={nc.TRUELAT1} +lat_2={nc.TRUELAT2} +lat_0={nc.CEN_LAT} +lon_0={nc.CEN_LON} +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs')
        elif MAP_PROJ == 2:
            # https://wrf-python.readthedocs.io/en/develop/user_api/index.html#projection-subclasses
            wrf_proj = wrf.PolarStereographic(truelat1=nc.TRUELAT1, stand_lon=nc.STAND_LON)
            proj4_str = wrf_proj.proj4()
            wrf_crs = pyproj.CRS.from_string(proj4_str)
        else:
            raise ValueError('WRF MAP_PROJ={} must be 1 or 2 (Lambert Concical LCC or Stereographic projection in PROJ.4).  Other values of MAP_PROJ are not supported'.format(MAP_PROJ))


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
        ## Projected coordinate space is north-up; whereas raster space (for WRF) is north-down.
        ## Because those two are different, the 0.5 needs to have positive sign.
        ## (If the yare the same, then we would use -0.5*dy)
        # (The above advice turned out to be wrong for WRF, it created an off-by-one error in the Y direction 2025-05-01)
        np.mean(yy_m_wrf[0,:]) - .5*dy,    # edge of y-coord of origin (most southerly pixel in this case)
        0,                        # Column rotation

        dy]                       # N-S pixel resolution (negative val for north-up images)

    wrf_grid = gisutil.RasterInfo(wrf_crs.to_wkt(), nji[1], nji[0], gt_wrf)

    # -----------------------------------------------------
    # Now check that the grid can reproduce the lat/lon values
    xy_x,xy_y = np.meshgrid(wrf_grid.centersx, wrf_grid.centersy)
    proj = pyproj.Proj(wrf_grid.wkt) 
    xy_lon, xy_lat = proj.transform(xy_x, xy_y, direction=pyproj.enums.TransformDirection.INVERSE)
    max_err_lon = np.max(np.abs(xy_lon - lon_m))
    max_err_lat = np.max(np.abs(xy_lat - lat_m))

    # Lon/lat values in the files are to single precision (7 digits), between 0-180.
    # Check that we match (almost) to that precision
    if max(max_err_lon, max_err_lat) > 1.e-4:
        raise ValueError(f'Something is wrong with setting up the WRF grid... the given grid does not match the original lon/lat values!  Maximum error is {max_error_lon} degrees in longitude and {max_err_lat} in latitude')

    return wrf_grid

def read_raw(data_fname, vname, units=None, fill_holes=False, keep_time=True):
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

    # Convert units to glaciological units if needed
    val = data_rawunits if units is None else cfutil.convert(data_rawunits, orig_units, units)

    # Get rid of Time dimension
    if (not keep_time) and len(val.shape) == 3:
        val = val[0,:]
    return val,nodata_value

#    if units is None:
#        return data_rawunits,nodata_value
#    val = cfutil.convert(data_rawunits, orig_units, units),nodata_value







def read(data_fname, vname, geo_fname, units=None, **kwargs):
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
    data,nodata_value = read_raw(data_fname, vname, units=units, **kwargs)
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
