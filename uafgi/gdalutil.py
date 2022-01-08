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

# @functional.memoize
# Do not memoize this.  Adding memoization produces the error when pickling:
#    Can't pickle <class 'uafgi.gdalutil.FileInfo'>: it's not the same object as uafgi.gdalutil.FileInfo
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
                self.nx = len(xx)
                self.dx = abs(xx[1]-xx[0])
                self.x = make_axis(xx)

                yy = nc.variables['y'][:].data
                self.yunits = nc.variables['y'].units
                self.ny = len(yy)
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
                        
                        GT = self.geotransform
                        a = GT[1]
                        b = GT[2]
                        c = GT[4]
                        d = GT[5]
                        det = a * d - b * c
                        bydet = 1./det
                        # Inverse matrix: (a,b,c,d)
                        self.geoinv = (d*bydet, -b*bydet, -c*bydet, a*bydet)

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

def clone_geometry(drivername, filename, grid_info, nBands, eType):
    """Creates a new dataset, based on the geometry of an existing raster
    file.

    drivername:
        Name of GDAL driver used to create dataset
    filename:
        Filename for dataset (or '' if driver type 'MEM')
    grid_info: GeoGrid (a Duck Type; for implementations grep for GeoGrid)
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


# NOTE: gdal_rasterize requires shapefile in meters, not degrees.
# And it requires the shapefile and raster CRS to be the same.
# ===> Not able to rasterize to spherical rasters
# See: Python Geospatial Analysis Cookbook
#      By Michael Diener
#      p. 60

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

    # http://www2.geog.ucl.ac.uk/~plewis/geogg122_current/_build/html/ChapterX_GDAL/OGR_Python.html

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

    mask_arr = dst_ds.GetRasterBand(1).ReadAsArray()
    return mask_arr



