import json
from gdal import ogr,osr
from uafgi import cdoutil,ncutil,functional
import gdal
import numpy as np

@functional.memoize
def gdal_srs():
    """GeoJSON files are all EPSG 4326 (lon/lat coordinates)"""

    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(4326)
    inSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return inSpatialRef


def iter_features(trace_file):
    """Iterate through featuress of a GeoJSON file (in lon/lat coordinates).
    trace_files:
        Name of GeoJSON file to read.
    """
    # https://stackoverflow.com/questions/42753745/how-can-i-parse-geojson-with-python
    with open(trace_file) as fin:
        gj = json.load(fin)

        assert gj['type'] == 'FeatureCollection'

        for feature in gj['features']:
            yield feature

def iter_traces(trace_file, proj):
    """Iterate through traces of a GeoJSON file and project to a local coordinate system.
    trace_files:
        GeoJSON filename
    proj:
        Converter from lon/lat to x/y
    """
    for feature in iter_features(trace_file):
        sdate = feature['properties']['date']
        date = datetime.datetime.fromisoformat(sdate).date()

        gline_lonlat = feature['geometry']['coordinates']
        gline_xx,gline_yy = proj.transform(
            np.array([x[0] for x in gline_lonlat]),
            np.array([x[1] for x in gline_lonlat]))

        yield date,(gline_xx,gline_yy)


# (formerly geojson_converter)
def get_tocf_proj(cf_file):

    """Creates a projection (via PROJ library) that converts from geojson
    lon/lat coordinates to coordinates derived from a CF-compliant NetCDF file

    NOTE: Currently only works if the variable "polar_stereographic" exists.

    cf_file: netCDF4.Dataset or filename
        File from which to fetch the destination projection
    """

    map_crs = cfutil.get_crs(cf_file)

    # Debugging
    # with open('crs.wkt', 'w') as fout:
    #    fout.write(wks_s)

    # Standard GeoJSON Coordinate Reference System (CRS)
    # Same as epsg:4326, but the urn: string is preferred
    # http://wiki.geojson.org/Rethinking_CRS
    # This CRS is lat/lon, whereas GeoJSON is lon/lat.  Use always_xy to fix that (below)
    geojson_crs = pyproj.CRS.from_string('urn:ogc:def:crs:OGC::CRS84')

    # https://pyproj4.github.io/pyproj/dev/examples.html
    # Note that crs_4326 has the latitude (north) axis first

    # Converts from geojson_crs to map_crs
    # See for always_xy: https://proj.org/faq.html#why-is-the-axis-ordering-in-proj-not-consistent
    proj = pyproj.Transformer.from_crs(geojson_crs, map_crs, always_xy=True)

    return proj

def check_error(err):
    if err != 0:
        raise 'GDAL Error {}'.format(err)


# https://pcjericks.github.io/py-gdalogr-cookbook/projection.html
def xload_layer(polygon_file, gridfile):

##    driver = gdal.GetDriverByName('netCDF')
#    grid_ds = gdal.Open(gridfile, gdal.GA_ReadOnly)
#    grid_proj_wkt = grid_ds.GetProjection()

    with ncutil.reopen_nc(gridfile) as nc:
        grid_proj_wkt = nc.variables['polar_stereographic'].spatial_ref

    grid_srs = osr.SpatialReference(wkt=grid_proj_wkt)
    print('gridfile ',gridfile)
    print('grid_srs ',grid_proj_wkt)

    driver = ogr.GetDriverByName('GeoJSON')
    src_ds = driver.Open(polygon_file)
    src_lyr = src_ds.GetLayer()   # Put layer number or name in her

    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(4326)
    inSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

#    outSpatialRef = osr.SpatialReference()
#    outSpatialRef.ImportFromEPSG(3413)
#    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, grid_srs)

# https://github.com/OSGeo/gdal/issues/1546
# SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER)

#    dst_ds = ogr.GetDriverByName('MEM').CreateDataSource('')
    dst_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource('x.shp')
    dst_lyr = dst_ds.CreateLayer('thelayer', srs=grid_srs)
    dst_lyr_def = dst_lyr.GetLayerDefn()

    while True:
        in_feature = src_lyr.GetNextFeature()
        if in_feature is None:
            break

        geom = in_feature.GetGeometryRef()
        geom.Transform(coordTrans)
        dst_feature = ogr.Feature(dst_lyr_def)
        dst_feature.SetGeometry(geom)
        dst_lyr.CreateFeature(dst_feature)
        dst_feature = None

        






# https://pcjericks.github.io/py-gdalogr-cookbook/projection.html
def load_layer(polygon_file, gridfile):

##    driver = gdal.GetDriverByName('netCDF')
#    grid_ds = gdal.Open(gridfile, gdal.GA_ReadOnly)
#    grid_proj_wkt = grid_ds.GetProjection()

    with ncutil.reopen_nc(gridfile) as nc:
        grid_proj_wkt = nc.variables['polar_stereographic'].spatial_ref

    grid_srs = osr.SpatialReference(wkt=grid_proj_wkt)
    print('gridfile ',gridfile)
    print('grid_srs ',grid_proj_wkt)

    driver = ogr.GetDriverByName('GeoJSON')
    src_ds = driver.Open(polygon_file)
    src_lyr = src_ds.GetLayer()   # Put layer number or name in her

    inSpatialRef = gdal_srs()


#    outSpatialRef = osr.SpatialReference()
#    outSpatialRef.ImportFromEPSG(3413)
#    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, grid_srs)

# https://github.com/OSGeo/gdal/issues/1546
# SetAxisMappingStrategy(OAMS_TRADITIONAL_GIS_ORDER)

#    dst_ds = ogr.GetDriverByName('MEM').CreateDataSource('')
    dst_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource('x.shp')
    dst_lyr = dst_ds.CreateLayer('thelayer', srs=grid_srs)
    dst_lyr_def = dst_lyr.GetLayerDefn()

    while True:
        in_feature = src_lyr.GetNextFeature()
        if in_feature is None:
            break

        geom = in_feature.GetGeometryRef()
        geom.Transform(coordTrans)
        dst_feature = ogr.Feature(dst_lyr_def)
        dst_feature.SetGeometry(geom)
        dst_lyr.CreateFeature(dst_feature)
        dst_feature = None

    return dst_ds



def rasterize_polygon(polygon_file, gridfile, tdir):
    """Generator yields rasterized version of polygons from polygon_file.

    polygon_feature:
        A feature of type polygon, as read from iter_features()
    gridfile:
        Name of NetCDF file containing projection, x, y etc. variables of local grid.
        Fine if it also contains data.
    Yields:
        Each specified layer in the polygon_file, rasterized
    """
    src_ds = load_layer(polygon_file, gridfile)
    src_lyr = src_ds.GetLayer()   # Put layer number or name in her

    print(src_lyr, src_lyr.GetExtent())
    fb = cdoutil.FileInfo(gridfile)

    dst_ds = gdal.GetDriverByName('netCDF').Create('x.nc', int(fb.nx), int(fb.ny), 1 ,gdal.GDT_Byte)
#    dst_ds = gdal.GetDriverByName('MEM').Create('', int(fb.nx), int(fb.ny), 1 ,gdal.GDT_Byte)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(0) #initialise raster with zeros
    dst_rb.SetNoDataValue(0)
    dst_ds.SetGeoTransform(fb.geotransform)

    maskvalue = 1
    check_error(gdal.RasterizeLayer(dst_ds, [1], src_lyr, burn_values=[maskvalue]))

    dst_ds.FlushCache()

    mask_arr=np.flipud(dst_ds.GetRasterBand(1).ReadAsArray())
    return mask_arr


def xxx(yyy):

    src_ds = ogr.Open(polygon_file)
    print(src_ds)

    return


    for feature in iter_feature(polygon_file):
        print(feature)
    return


    fb = cdoutil.FileInfo(gridfile)


    for fid in fids:

        # Select single feature into a gjfile
        # https://gis.stackexchange.com/questions/330811/how-to-rasterize-individual-feature-polygon-from-gjfile-using-gdal-ogr-in-p
        one_shape = tdir.join('one_shape.shp')
        select_feature(gjfile, fid, one_shape)
        print(pathlib.Path(one_shape).stat().st_size)

        src_ds = ogr.Open(one_shape)
        src_lyr = src_ds.GetLayer()   # Put layer number or name in her

#        dst_ds = gdal.GetDriverByName('netCDF').Create('x{}.nc'.format(fid), int(fb.nx), int(fb.ny), 1 ,gdal.GDT_Byte)
        dst_ds = gdal.GetDriverByName('MEM').Create('', int(fb.nx), int(fb.ny), 1 ,gdal.GDT_Byte)
        dst_rb = dst_ds.GetRasterBand(1)
        dst_rb.Fill(0) #initialise raster with zeros
        dst_rb.SetNoDataValue(0)
        dst_ds.SetGeoTransform(fb.geotransform)

        check_error(gdal.RasterizeLayer(dst_ds, [1], src_lyr, burn_values=[maskvalue]))

        dst_ds.FlushCache()

        mask_arr=np.flipud(dst_ds.GetRasterBand(1).ReadAsArray())
        yield mask_arr
