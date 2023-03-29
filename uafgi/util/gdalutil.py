import json,subprocess
import collections
import numpy as np
import netCDF4, cf_units
from uafgi.util import cfutil,ncutil,gisutil
from osgeo import osr,gdal

def check_error(err):
    """Checks error code return from GDAL functions, and raises an
    exception if needed."""

    if err != 0:
        raise GDALException('GDAL Error {}'.format(err))


# -------------------------------------------------------

# -----------------------------------------------------------------
def is_netcdf_file(fname):
    try:
        with open(fname, 'r') as nc:
            return True
    except:
        return False

def file_info(raster_file):
    """Top level user function"""

    # Special case for NetCDF
    if is_netcdf_file(raster_file):
        return NCRasterInfo(raster_file)

    # ----------------------------------------
    # Not NetCDF, just do general GDAL
    # TODO: See here how to do the same thing with core GDAL Python calls
    # https://gdal.org/user/raster_data_model.html

    if False:
        # Get the raw data
        cmd = ['gdalinfo', '-json', raster_file]
        js = json.loads(subprocess.check_output(cmd))

        # I only know how to properly interpret "Area" files
        md = js['metadata']['']    # For some reason it's a level down
        assert md['AREA_OR_POINT'] == 'Area'

        return gisutil.RasterInfo(js['coordinateSystem']['wkt'], js['size'][0], js['size'][1], js['geoTransform'])
    else:
        #info = gdal.Info(raster_file, format='json')

        # https://drr.ikcest.org/tutorial/k8022
        ds = gdal.Open(raster_file)
        return gisutil.RasterInfo(
            ds.GetProjection(),
            ds.RasterXSize, ds.RasterYSize,
            raster.GetGeoTransform())


def grid_info(raster_file):
    ds = gdal.Open(raster_file)
    grid_info = gisutil.RasterInfo(
        ds.GetProjection(),
        ds.RasterXSize, ds.RasterYSize,
        np.array(ds.GetGeoTransform()))
    return grid_info

def read_raster(raster_file):
    """Simple way to read a raster file; and return it as a Numpy Array.
    Assumes single-band raster files (the usual case)

    Returns: grid_info, data
        grid_info: gisutil.RasterInfo
            Description of the raster file's grid
        data: np.array
            Data found in the raster file."""

    ds = gdal.Open(raster_file)
    grid_info = gisutil.RasterInfo(
        ds.GetProjection(),
        ds.RasterXSize, ds.RasterYSize,
        np.array(ds.GetGeoTransform()))
    band = ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    data = band.ReadAsArray()
    return grid_info, data, nodata_value


def write_raster(raster_file, grid_info, data, nodata_value, driver='GTiff', type=gdal.GDT_Float64, options=['COMPRESS=LZW']):
    """
    type:
        One of Byte, UInt16, Int16, UInt32, Int32, UInt64, Int64,
        Float32, Float64, and the complex types CInt16, CInt32,
        CFloat32, and CFloat64.
        Must match the datatype of the numpy array data
        Eg: gdal.GDT_Byte
    """
    # https://gis.stackexchange.com/questions/351970/python-gdal-write-a-geotiff-in-colour-from-a-data-array

    # Open output file
    driver_obj = gdal.GetDriverByName(driver)
    dst_ds = driver_obj.Create(raster_file, grid_info.nx, grid_info.ny, 1, type, options=options)

    # Set the CRS
    dst_ds.SetProjection(grid_info.wkt)
    dst_ds.SetGeoTransform(list(grid_info.geotransform))

    # Store the data
    rb = dst_ds.GetRasterBand(1)
    if nodata_value is not None:
        rb.SetNoDataValue(nodata_value)
    rb.WriteArray(data)

# -----------------------------------------------------------------
def clone_geometry(drivername, filename, grid_info, nBands, eType):
    """Creates a new dataset, based on the geometry of an existing raster
    file.

    drivername:
        Name of GDAL driver used to create dataset
    filename:
        Filename for dataset to create (or '' if driver type 'MEM')
    grid_info: GeoGrid (a Duck Type; for implementations grep for GeoGrid)
        Result of RasterInfo() from an existing raster file
    nBands:
        Number of bands
    eType:
        type of raster (eg gdal.GDT_Byte)
    https://gdal.org/api/gdaldriver_cpp.html

    """

    driver = gdal.GetDriverByName(drivername)
    ds = driver.Create(filename, grid_info.nx, grid_info.ny, nBands, eType)
    srs = osr.SpatialReference(wkt=grid_info.wkt)
    ds.SetSpatialRef(srs)
    ds.SetGeoTransform(grid_info.geotransform)
    return ds


# NOTE: gdal_rasterize requires shapefile in meters, not degrees.
# And it requires the shapefile and raster CRS to be the same.
# ===> Not able to rasterize to spherical rasters
# See: Python Geospatial Analysis Cookbook
#      By Michael Diener
#      p. 60






# ======================================================================================
# Legacy Code

Axis = collections.namedtuple('Axis', (
    'centers',    # Center of each pixel on the axis
    'n',          # Number of pixels in centers
    'low', 'high', # Range of the axis to the EDGES of the pixels
    'delta',     # Size of pixels
    'low_raw', 'high_raw', 'delta_raw'
))

def make_axis(centers):
    if centers[1] > centers[0]:
        # Positive axis
        dx = centers[1]-centers[0]
        half_dx = .5 * dx
        return Axis(
            centers, len(centers),
            centers[0] - half_dx,
            centers[-1] + half_dx,
            dx,
            centers[0] - half_dx,
            centers[-1] + half_dx,
            dx)

    else:
        # Negative axis; but dx should still be positive
        dx = centers[0]-centers[1]
        half_dx = .5 * dx
        return Axis(
            centers, len(centers),
            centers[-1] - half_dx,
            centers[0] + half_dx,
            dx,
            centers[0] - half_dx,
            centers[-1] + half_dx,
            -dx)

# @functional.memoize
# Do not memoize this.  Adding memoization produces the error when pickling:
#    Can't pickle <class 'uafgi.gisutil.RasterInfo'>: it's not the same object as uafgi.gisutil.RasterInfo
class NCRasterInfo(object):
    """Reads spatial extents from netCDF raster file.
    This is "legacy," but also robust to many types of incomplete NetCDF files.

    May be used, eg, as:
                '-projwin', str(x0), str(y1), str(x1), str(y0),
                '-tr', str(dx), str(dy),
    Returns:
        self.x0, self.x1:
            Min, max of region in the file
        self.dx:
            Grid spacing in x direction
        self.srs: osr.SpatialReference
            GDAL Coordinate reference system (CRS) used in the file
        geotransform: list
            GDAL domain used in this file
    """
    def __init__(self, grid_file):
        """Obtains bounding box of a grid; and also the time dimension, if it exists.
        Returns: x0,x1,y0,y1
        """

        with ncutil.open(grid_file) as nc:
            # Info on spatial bounds
            if 'x' in nc.variables:
                xx = nc.variables['x'][:].data
                self.xunits = nc.variables['x'].units
#                self.nx = len(xx)
                self.dx = abs(xx[1]-xx[0])
                self.x = make_axis(xx)

                yy = nc.variables['y'][:].data
                self.yunits = nc.variables['y'].units
#                self.ny = len(yy)
                self.dy = abs(yy[1]-yy[0])
                self.y = make_axis(yy)

                # Info on the coordinate reference system (CRS)
                if 'polar_stereographic' in nc.variables:
                    ncv = nc.variables['polar_stereographic']
                    self.wkt = ncv.spatial_ref

                    if hasattr(ncv, 'GeoTransform'):
                        sgeotransform = ncv.GeoTransform
                        self.geotransform = tuple(float(x) for x in sgeotransform.split(' ') if len(x) > 0)

                        # The GeoTransform is a tuple of 6 values,
                        # which relate raster indices into
                        # coordinates.
                        # Xgeo = GT[0] + Xpixel*GT[1] + Yline*GT[2]
                        # Ygeo = GT[3] + Xpixel*GT[4] + Yline*GT[5]
                        
                        self.geoinv = invert_geotransform(self.geotransform)

            # Info on time units
            if 'time' in nc.variables:
                nctime = nc.variables['time']

                # Times in original form
                self.time_units = cf_units.Unit(nctime.units, nctime.calendar)
                self.times = nc.variables['time'][:]    # "days since <refdate>

                # Convert to Python datetimes
                self.datetimes = [self.time_units.num2date(t_d)
                    for t_d in self.times]

                # Convert to times in "seconds since <refdate>"
                self.time_units_s = cfutil.replace_reftime_unit(
                    self.time_units, 'seconds')
                self.times_s = [self.time_units.convert(t_d, self.time_units_s)
                    for t_d in self.times]

    # -----------------------------------------------------
    # Implement the "GeoGrid" Duck Typing API
    # See also: cfutil.LonlatGrid
    @property
    def nx(self):
        return self.x.n

    @property
    def ny(self):
        return self.y.n

    # self.geotransform is already a property

    @property
    def srs(self):
        # Use srs.ExportToWkt() to get back to Wkt string
        return osr.SpatialReference(wkt=self.wkt)
    # -----------------------------------------------------

    @property
    def extents(self):
        """Provide extents for Cartopy / Matplotlib's ax.set_extent()"""
        gt = self.geotransform
        x0 = gt[0]
        x1 = x0 + gt[1] * self.nx

        y0 = gt[3]
        y1 = y0 + gt[5] * self.ny


        return [x0,x1,y0,y1]


    def to_xy(self, i, j):
        """Converts an (i,j) pixel address to an (x,y) geographic value"""
        GT = self.geotransform
        Xgeo = GT[0] + i*GT[1] + j*GT[2]
        Ygeo = GT[3] + j*GT[4] + j*GT[5]
        return Xgeo,Ygeo


    def to_ij(self, x, y):
        """Converts an (x,y) value to an (i,j) index into raster
        NOTE: Similar to geoio.GeoImage.proj_to_raster()"""

        # https://stackoverflow.com/questions/40464969/why-does-gdal-grid-turn-image-upside-down
        xx = (x - self.geotransform[0])
        yy = (y - self.geotransform[3])
        ir = self.geoinv[0]*xx + self.geoinv[1]*yy
        jr = self.geoinv[2]*xx + self.geoinv[3]*yy

        return int(ir+.5), int(jr+.5)
# ======================================================================================

