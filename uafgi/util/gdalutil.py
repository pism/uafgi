import json,subprocess,typing,pathlib
import collections
import numpy as np
import netCDF4, cf_units
from uafgi.util import cfutil,ncutil,gisutil
from osgeo import osr,gdal,gdal_array
from osgeo import gdalconst

def file_in_zip(zip_file, arcname):

    """Creates a string that will cause GDAL/OGR to read a raster or
    vector file out of a zip file.
    zip_file:
        Name of the container zip file
    arcname:
        Name of the file within the zip container
    """
    return '/vsizip/{}/{}'.format(str(zip_file), arcname)


def resolve_file(file):
    """
    Resolves (zip_file, arcname) to file_in_zip(zip_file, arcname)
    """
    if isinstance(file, str) or isinstance(file, pathlib.PurePath):
        return file

    if file[0].parts[-1].endswith('.zip'):
        return file_in_zip(*file)

    raise ValueError(f'Cannot resolve filename: {file}')

# -------------------------------------------------------------------
def positive_rectangle(x0,x1,y0,y1):
    """Returns a rectangle, in which x1>x0 and y1>y0, even if it didn't start out that way."""
    if x1<x0:
        x0,x1 = x1,x0
    if y1<y0:
        y0,y1 = y1,y0
    return (x0,x1,y0,y1)

# -------------------------------------------------------------------
#>>> dvs = [gdal.GetDriver(i).ShortName for i in range(gdal.GetDriverCount())]
#>>> dvs
#['VRT', 'DERIVED', 'GTiff', 'COG', 'NITF', 'RPFTOC', 'ECRGTOC', 'HFA', 'SAR_CEOS', 'CEOS', 'JAXAPALSAR', 'GFF', 'ELAS', 'ESRIC', 'AIG', 'AAIGrid', 'GRASSASCIIGrid', 'ISG', 'SDTS', 'DTED', 'PNG', 'JPEG', 'MEM', 'JDEM', 'GIF', 'BIGGIF', 'ESAT', 'FITS', 'BSB', 'XPM', 'BMP', 'DIMAP', 'AirSAR', 'RS2', 'SAFE', 'PCIDSK', 'PCRaster', 'ILWIS', 'SGI', 'SRTMHGT', 'Leveller', 'Terragen', 'netCDF', 'HDF4', 'HDF4Image', 'ISIS3', 'ISIS2', 'PDS', 'PDS4', 'VICAR', 'TIL', 'ERS', 'JP2OpenJPEG', 'L1B', 'FIT', 'GRIB', 'RMF', 'WCS', 'WMS', 'MSGN', 'RST', 'GSAG', 'GSBG', 'GS7BG', 'COSAR', 'TSX', 'COASP', 'R', 'MAP', 'KMLSUPEROVERLAY', 'WEBP', 'PDF', 'Rasterlite', 'MBTiles', 'PLMOSAIC', 'CALS', 'WMTS', 'SENTINEL2', 'MRF', 'TileDB', 'PNM', 'DOQ1', 'DOQ2', 'PAux', 'MFF', 'MFF2', 'GSC', 'FAST', 'BT', 'LAN', 'CPG', 'NDF', 'EIR', 'DIPEx', 'LCP', 'GTX', 'LOSLAS', 'NTv2', 'CTable2', 'ACE2', 'SNODAS', 'KRO', 'ROI_PAC', 'RRASTER', 'BYN', 'ARG', 'RIK', 'USGSDEM', 'GXF', 'KEA', 'BAG', 'HDF5', 'HDF5Image', 'NWT_GRD', 'NWT_GRC', 'ADRG', 'SRP', 'BLX', 'PostGISRaster', 'SAGA', 'XYZ', 'HF2', 'OZI', 'CTG', 'ZMap', 'NGSGEOID', 'IRIS', 'PRF', 'EEDAI', 'EEDA', 'DAAS', 'SIGDEM', 'TGA', 'OGCAPI', 'STACTA', 'STACIT', 'GNMFile', 'GNMDatabase', 'ESRI Shapefile', 'MapInfo File', 'UK .NTF', 'LVBAG', 'OGR_SDTS', 'S57', 'DGN', 'OGR_VRT', 'Memory', 'CSV', 'NAS', 'GML', 'GPX', 'LIBKML', 'KML', 'GeoJSON', 'GeoJSONSeq', 'ESRIJSON', 'TopoJSON', 'Interlis 1', 'Interlis 2', 'OGR_GMT', 'GPKG', 'SQLite', 'WAsP', 'PostgreSQL', 'OpenFileGDB', 'DXF', 'CAD', 'FlatGeobuf', 'Geoconcept', 'GeoRSS', 'VFK', 'PGDUMP', 'OSM', 'GPSBabel', 'OGR_PDS', 'WFS', 'OAPIF', 'EDIGEO', 'SVG', 'Idrisi', 'XLS', 'ODS', 'XLSX', 'Elasticsearch', 'Carto', 'AmigoCloud', 'SXF', 'Selafin', 'JML', 'PLSCENES', 'CSW', 'VDV', 'GMLAS', 'MVT', 'NGW', 'MapML', 'TIGER', 'AVCBin', 'AVCE00', 'GenBin', 'ENVI', 'EHdr', 'ISCE', 'Zarr', 'HTTP']

# -----------------------------------------------------------------
# NOTE on using Numpy arrays as GDAL Data Sources
# Frank Warmerdam warmerda (Wed Jul 19 18:37:02 EDT 2000)
# https://lists.osgeo.org/pipermail/gdal-dev/2000-July/002975.html
# 
# GDAL supports NumPy arrays as a special format. The filename passed
# to the gdal.Open() call should be "NUMPY:::0xhhhhhhhh" with the h's
# containing the hex pointer value of the python array object. The
# gdalnumeric.GetArrayFilename() function can be used to create this
# for any array.
#
# The function gdal_array.OpenArray() uses GetArrayFilename() as so.
# Apparently, OpenArray() requires prototype_ds, which is uses to
# obtain georeference info.
#
# def OpenArray( array, prototype_ds = None ):
# 
#     ds = gdal.Open( GetArrayFilename(array) )
# 
#     if ds is not None and prototype_ds is not None:
#         if type(prototype_ds).__name__ == 'str':
#             prototype_ds = gdal.Open( prototype_ds )
#         if prototype_ds is not None:
#             CopyDatasetInfo( prototype_ds, ds )
#             
#     return ds
#
# This seems to be an alternative, using the MEM driver:
# https://gis.stackexchange.com/questions/75891/in-python-reading-a-gdal-raster-from-memory-instead-of-a-file
# from osgeo import gdal
# import numpy as np
# driver = gdal.GetDriverByName('MEM')
# src_ds = driver.Create('', 100, 200, 1)
# band = src_ds.GetRasterBand(1)
# # Create random data array to put into the raster object
# ar = np.random.randint(0, 255, (200, 100))
# band.WriteArray(ar)
# -----------------------------------------------------------------
def check_error(err):
    """Checks error code return from GDAL functions, and raises an
    exception if needed."""

    if err != 0:
        raise GDALException('GDAL Error {}'.format(err))
# -----------------------------------------------------------------
def is_netcdf_file(fname):
    try:
        with open(fname, 'r') as nc:
            return True
    except:
        return False

#def file_info(raster_file):
#    """Top level user function"""
#
#    # Special case for NetCDF
#    if is_netcdf_file(raster_file):
#        return NCRasterInfo(raster_file)
#
#    # ----------------------------------------
#    # Not NetCDF, just do general GDAL
#    # TODO: See here how to do the same thing with core GDAL Python calls
#    # https://gdal.org/user/raster_data_model.html
#
#    if False:
#        # Get the raw data
#        cmd = ['gdalinfo', '-json', raster_file]
#        js = json.loads(subprocess.check_output(cmd))
#
#        # I only know how to properly interpret "Area" files
#        md = js['metadata']['']    # For some reason it's a level down
#        assert md['AREA_OR_POINT'] == 'Area'
#
#        return gisutil.RasterInfo(js['coordinateSystem']['wkt'], js['size'][0], js['size'][1], js['geoTransform'])
#    else:
#        #info = gdal.Info(raster_file, format='json')
#
#        # https://drr.ikcest.org/tutorial/k8022
#        ds = gdal.Open(raster_file)
#        return gisutil.RasterInfo(
#            ds.GetProjection(),
#            ds.RasterXSize, ds.RasterYSize,
#            raster.GetGeoTransform())


def grid_info(raster_file):
    """Extracts a RasterInfo from a GAL-openable raster file"""
    ds = gdal.Open(str(raster_file))
    grid_info = gisutil.RasterInfo(
        ds.GetProjection(),
        ds.RasterXSize, ds.RasterYSize,
        np.array(ds.GetGeoTransform()))
    return grid_info

def read_grid(raster_file):
    ds = gdal.Open(str(raster_file))
    grid_info = gisutil.RasterInfo(
        ds.GetProjection(),
        ds.RasterXSize, ds.RasterYSize,
        np.array(ds.GetGeoTransform()))
    return grid_info


class Raster(typing.NamedTuple):
    grid: object
    data: object
    nodata: object    # Nodata value


def read_raster(raster_file, data=True):
    """Simple way to read a raster file; and return it as a Numpy Array.
    Assumes single-band raster files (the usual case)

    data:
        Read the data portion of the raster?

    Returns: grid_info, data, nodata_value
        grid_info: gisutil.RasterInfo
            Description of the raster file's grid
        data: np.array
            Data found in the raster file."""

    raster_file = resolve_file(raster_file)
    ds = gdal.Open(str(raster_file))
    grid_info = gisutil.RasterInfo(
        ds.GetProjection(),
        ds.RasterXSize, ds.RasterYSize,
        np.array(ds.GetGeoTransform()))
    band = ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    _data = band.ReadAsArray() if data else None
    return Raster(grid_info, _data, nodata_value)

def raster_ds(raster):
    """Constructs an in-memory dataset from an in-memory array"""
    grid_info, data, nodata_value = raster

    ds = gdal_array.OpenArray(data)    # returns None on error
    set_grid_info(ds, grid_info, nodata_value)    # nodata_value must be of appropriate type
    return ds


def write_raster(raster_file, grid_info, data, nodata_value, driver='GTiff', type=gdal.GDT_Float64, options=['COMPRESS=LZW', 'TFW=YES'], metadata=None):
    """
    type:
        One of Byte, UInt16, Int16, UInt32, Int32, UInt64, Int64,
        Float32, Float64, and the complex types CInt16, CInt32,
        CFloat32, and CFloat64.
        Must match the datatype of the numpy array data
        Eg: gdal.GDT_Byte
    metadata: (OPTIONAL) {str: str, ...} or [(str, str), ...]
        Dict of metadata to store in file (key/value pairs)
    """
    # https://gis.stackexchange.com/questions/351970/python-gdal-write-a-geotiff-in-colour-from-a-data-array

    # Open output file
    driver_obj = gdal.GetDriverByName(driver)
    dst_ds = driver_obj.Create(str(raster_file), grid_info.nx, grid_info.ny, 1, type, options=options)

    # Set the CRS
    dst_ds.SetProjection(grid_info.wkt)
    dst_ds.SetGeoTransform(list(grid_info.geotransform))
    if metadata is not None:
        dst_ds.SetMetadata(metadata)

    # Store the data
    rb = dst_ds.GetRasterBand(1)
    if nodata_value is not None:
        rb.SetNoDataValue(np.double(nodata_value))
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
# -------------------------------------------------------
def set_grid_info(ds, grid_info, nodata_value=None):
    """
    ds:
        Data Source to set up with reference info
    grid_info: gisutil.RasterInfo
        Reference info
    """
    ds.SetProjection(grid_info.wkt)
    ds.SetGeoTransform(list(grid_info.geotransform))
    rb = ds.GetRasterBand(1)
    if nodata_value is not None:
#        print('Setting nodata_value of ', nodata_value, type(nodata_value))
        rb.SetNoDataValue(nodata_value)
    #ds.SetNoDataValue(nodata_value)

# -------------------------------------------------------
def regrid(idata, igrid_info, inodata_value, ogrid_info, onodata_value, resample_algo=gdalconst.GRA_NearestNeighbour):
    """Regrids a Numpy array from one CRS/etc to another.
    igrid_info: gisutil.RasterInfo
    units: str (OPTIONAL)
        Convert to these units upon read
    Returns: np.array
    """

    # Construct an in-memory dataset for the input grid info
    ids = gdal_array.OpenArray(idata)    # returns None on error
    print('Opening of type ', idata.dtype)
#    set_grid_info(ids, igrid_info, idata.dtype.type(inodata_value))
    set_grid_info(ids, igrid_info, np.double(inodata_value))

    # Construct an in-memory dataset for the output grid info
    odata = np.zeros((ogrid_info.ny, ogrid_info.nx))
    ods = gdal_array.OpenArray(odata)
    ods.GetRasterBand(1).SetNoDataValue(onodata_value)
    ods.GetRasterBand(1).Fill(onodata_value)
#    set_grid_info(ods, ogrid_info, idata.dtype.type(onodata_value))
#    set_grid_info(ods, ogrid_info, np.double(onodata_value))    # GDAL always wants C/SWIG double
    set_grid_info(ods, ogrid_info)    # GDAL always wants C/SWIG double

    # Regrid
    # https://stackoverflow.com/questions/10454316/how-to-project-and-resample-a-grid-to-match-another-grid-with-gdal-python

    # GRA_NearestNeighbour , GRA_Bilinear = 1 , GRA_Cubic = 2 , GRA_CubicSpline = 3 , 
    # GRA_Lanczos = 4 , GRA_Average = 5 , GRA_Mode = 6 , GRA_Max , 
    # GRA_Min , GRA_Med , GRA_Q1 = 11 , GRA_Q3 = 12 , 
    # GRA_Sum = 13 , GRA_RMS = 14 
    gdal.ReprojectImage(ids, ods, ids.GetProjection(), ods.GetProjection(), resample_algo)
    return odata
# -------------------------------------------------------1
# CPLErr GDALComputeProximity(GDALRasterBandH hSrcBand, GDALRasterBandH hProximityBand, char **papszOptions, GDALProgressFunc pfnProgress, void *pProgressArg)
# Compute the proximity of all pixels in the image to a set of pixels in the source image.
# 
# This function attempts to compute the proximity of all pixels in the image to a set of pixels in the source image. The following options are used to define the behavior of the function. By default all non-zero pixels in hSrcBand will be considered the "target", and all proximities will be computed in pixels. Note that target pixels are set to the value corresponding to a distance of zero.
# 
# The progress function args may be NULL or a valid progress reporting function such as GDALTermProgress/NULL.
# 
# Options:
# 
# VALUES=n[,n]*
# 
# A list of target pixel values to measure the distance from. If this option is not provided proximity will be computed from non-zero pixel values. Currently pixel values are internally processed as integers.
# 
# DISTUNITS=[PIXEL]/GEO
# 
# Indicates whether distances will be computed in pixel units or in georeferenced units. The default is pixel units. This also determines the interpretation of MAXDIST.
# 
# MAXDIST=n
# 
# The maximum distance to search. Proximity distances greater than this value will not be computed. Instead output pixels will be set to a nodata value.
# 
# NODATA=n
# 
# The NODATA value to use on the output band for pixels that are beyond MAXDIST. If not provided, the hProximityBand will be queried for a nodata value. If one is not found, 65535 will be used.
# 
# USE_INPUT_NODATA=YES/NO
# 
# If this option is set, the input data set no-data value will be respected. Leaving no data pixels in the input as no data pixels in the proximity output.
# 
# FIXED_BUF_VAL=n
# 
# If this option is set, all pixels within the MAXDIST threadhold are set to this fixed value instead of to a proximity distance.

def compute_proximity(gridA, srcA, maxdist, src_nd=None):
    """
    srcA:
        integer array in gridA.  (Use .astype(int) to convert from a bool array)
        !=0: target cells ("ocean")
        0: non-target cells ("land")
    srcA_nd:
        Nodata value for srcA, if any nodata cells exist.
    """

    # Options for the ComputeProxmity() call
    # https://svn.osgeo.org/gdal/trunk/gdal/swig/python/scripts/gdal_proximity.py
    # https://gdal.org/api/gdal_alg.html#_CPPv420GDALComputeProximity15GDALRasterBandH15GDALRasterBandHPPc16GDALProgressFuncPv
    options = ['DISTUNITS=GEO', f'MAXDIST={maxdist}', f'NODATA={maxdist}']

    # Construct an in-memory dataset for the input grid info
    srcA_ds = gdal_array.OpenArray(srcA)    # returns None on error
    if src_nd is None:
        set_grid_info(srcA_ds, gridA)
    else:
        set_grid_info(srcA_ds, gridA, srcA_nd)
        options.append('USE_INPUT_NODATA=YES')
    srcA_rb = srcA_ds.GetRasterBand(1)

    # Constrcut output dataset
#    proxA = np.zeros(srcA.shape, dtype='d')
    proxA = np.zeros(srcA.shape, dtype=np.byte)
    proxA_ds = gdal_array.OpenArray(proxA)
    proxA_rb = proxA_ds.GetRasterBand(1)
    proxA_rb.SetNoDataValue(maxdist)


    options.append('FIXED_BUF_VAL=17')
    gdal.ComputeProximity(srcA_rb, proxA_rb, options, callback=gdal.TermProgress)

    return proxA
