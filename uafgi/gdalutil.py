import collections
import numpy as np
import netCDF4, cf_units
from uafgi import functional,ogrutil,cfutil,ncutil
from osgeo import osr,ogr,gdal

def check_error(err):
    """Checks error code return from GDAL functions, and raises an
    exception if needed."""

    if err != 0:
        raise GDALException('GDAL Error {}'.format(err))

def open(fname, driver=None, **kwargs):
    """Opens a GDAL datasource.  Raises exception if not found."""
    ds = ogr.GetDriverByName(driver).Open(fname, **kwargs)
    if ds is None:
        raise FileNotFoundException(fname)
    return ds

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

@functional.memoize
class FileInfo(object):
    """Reads spatial extents from GDAL raster file.
    Currently only works for NetCDF raster files.

    May be used, eg, as:
                '-projwin', str(x0), str(y1), str(x1), str(y0),
                '-tr', str(dx), str(dy),
    Returns:
        self.x0, self.x1:
            Min, max of region in the file
        self.dx:
            Grid spacing in x direction
        welf.srs: osr.SpatialReference
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
                self.x = make_axis(nc.variables['x'][:])
                self.y = make_axis(nc.variables['y'][:])

                # Info on the coordinate reference system (CRS)
                if 'polar_stereographic' in nc.variables:
                    ncv = nc.variables['polar_stereographic']
                    self.srs = osr.SpatialReference(wkt=ncv.spatial_ref)
                    # Use srs.ExportToWkt() to get back to Wkt string

                    if hasattr(ncv, 'GeoTransform'):
                        sgeotransform = ncv.GeoTransform
                        self.geotransform = tuple(float(x) for x in sgeotransform.split(' ') if len(x) > 0)

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

    def to_ij(self, x, y):
        """Converts an (x,y) value to an (i,j) index into raster"""
        i = int((x-self.x.low_raw) / self.x.delta_raw)
        j = int((y-self.y.low_raw) / self.x.delta_raw)
        return i,j

def clone_geometry(drivername, filename, grid_info, nBands, eType):
    """Creates a new dataset, based on the geometry of an existing raster
    file.

    drivername:
        Name of GDAL driver used to create dataset
    filename:
        Filename for dataset (or '' if driver type 'MEM')
    grid_info:
        Result of FileInfo() from an existing raster file
    nBands:
        Number of bankds
    eType:
        type of raster (eg gdal.GDT_Byte)
    https://gdal.org/api/gdaldriver_cpp.html

    """

    driver = gdal.GetDriverByName(drivername)
    ds = driver.Create(filename, grid_info.nx, grid_info.ny, nBands, eType)
    ds.SetSpatialRef(grid_info.srs)
    ds.SetGeoTransform(grid_info.geotransform)
    return ds



def rasterize_polygons(polygon_ds, grid_info):
    """Rasterizes all polygons from polygon_ds into a single raster, which
    is returned as a Numpy array.

    polygon_ds:
        Open GDAL dataset containing polygons in a single layer
        Can be Shapefile, GeoJSON, etc.
        Eg: poly_ds = gdalutil.open(outlines_shp, driver='ESRI Shapefile')

    grid_info: gdalutil.FileInfo
        Definition of the grid used for fjord
        Eg: gdalutil.FileInfo(grid_file)

    Returns: np.ndarray
        Mask equals 1 inside the polygons, and 0 outside.
    """

    # Reproject original polygon file to a new (internal) dataset
    # src_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource('x.shp')
    src_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    ogrutil.reproject(polygon_ds, grid_info.srs, src_ds)
    src_lyr = src_ds.GetLayer()   # Put layer number or name in here

    # Create destination raster dataset
#    dst_ds = clone_geometry('netCDF', 'x.nc', grid_info, 1,  gdal.GDT_Byte)
    dst_ds = clone_geometry('MEM', '', grid_info, 1,  gdal.GDT_Byte)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(0) #initialise raster with zeros
    dst_rb.SetNoDataValue(0)

    maskvalue = 1
    bands = [1]          # Bands to rasterize into
    burn_values = [1]    # Burn this value for each band
    check_error(gdal.RasterizeLayer(dst_ds, bands, src_lyr, burn_values=burn_values))

    dst_ds.FlushCache()

    mask_arr=np.flipud(dst_ds.GetRasterBand(1).ReadAsArray())
    return mask_arr



