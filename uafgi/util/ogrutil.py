import typing
from osgeo import ogr,osr
from uafgi.util import osrutil
import shapely        # Deprecated
import pandas as pd
import numpy as np

#def open(fname, driver=None, **kwargs):
#    """Opens a GDAL datasource.  Raises exception if not found."""
#    ds = ogr.GetDriverByName(driver).Open(fname, **kwargs)
#    if ds is None:
#        raise FileNotFoundException(fname)
#    return ds

def reproject(src_ds, dst_srs, dst_ds):
    """Reprojects an entire OGR dataset.

    src_ds:
        Source dataset, eg:
            driver = ogr.GetDriverByName('GeoJSON')
            src_ds = driver.Open(polygon_file)

    dst_srs: osr.SpatialReference
        Destination SRS to convert to

    dst_ds:
        Destination dataset, eg one of:
           dst_ds = ogr.GetDriverByName('MEM').CreateDataSource('')
           dst_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource('x.shp')
    """

    src_lyr = src_ds.GetLayer()   # Put layer number or name in her
    src_srs = src_lyr.GetSpatialRef()
    # If no coord system on source, assume it's OK and doesn't need transforming
    coordTrans = None if src_srs is None else osr.CoordinateTransformation(src_srs, dst_srs)

    dst_lyr = dst_ds.CreateLayer(src_lyr.GetName(), srs=dst_srs)
    dst_lyr_def = dst_lyr.GetLayerDefn()

    # Copy fields from source to destination
    src_feature = src_lyr.GetFeature(0)
    for i in range(src_feature.GetFieldCount()):
        dst_lyr.CreateField(src_feature.GetFieldDefnRef(i))

    while True:
        in_feature = src_lyr.GetNextFeature()
        if in_feature is None:
            break

        # Reproject the feature
        geom = in_feature.GetGeometryRef()
        if coordTrans is not None:
            geom.Transform(coordTrans)
        dst_feature = ogr.Feature(dst_lyr_def)
        dst_feature.SetGeometry(geom)

        # Copy the fields
        for i in range(0, dst_lyr_def.GetFieldCount()):
            dst_feature.SetField(dst_lyr_def.GetFieldDefn(i).GetNameRef(), in_feature.GetField(i))

        # Finish
        dst_lyr.CreateFeature(dst_feature)
        dst_feature = None

    return dst_ds

def to_shapely_polygon(feature, osr_transform):
    """Converts an OGR polygon (loaded as a feature) to a Shapely polygon.

    feature:
        An OGR feature.  Eg:

        driver = ogr.GetDriverByName('ESRI Shapefile')
        src_ds = driver.Open('troughs/shp/fjord_outlines.shp')
        src_lyr = src_ds.GetLayer()   # Put layer number or name in her
        while True:
            feat = src_lyr.GetNextFeature()
            if feat is None:
                break

    osr_transform:
        OSR-style transformer between two coordinate systems.  Eg:
            src_srs = src_lyr.GetSpatialRef()
            dst_srs.ImportFromWkt(the_wkt)
            osr_transform = osr.CoordinateTransformation(src_srs, dst_srs)
    """
    geom = feature.GetGeometryRef()
    geom.Transform(osr_transform)
    ring = geom.GetGeometryRef(0)
    npoints = ring.GetPointCount()
    points = list()
    for p in range(0,npoints):
        x,y,z = ring.GetPoint(p)
        points.append(shapely.geometry.Point(x,y))

    poly = shapely.geometry.Polygon(points)
    return poly


def to_srs(wkt):
    """Creates an OGR SRS object from a WKT string"""
    srs = ogr.osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return srs
# ================= MOST EVERYTHING ABOVE THIS LINE IS OBSOLETE =====================
# --------------------------------------------------------------------
class Shapefile(typing.NamedTuple):
    df: pd.DataFrame    # Data from the shapefile (including shape col, if present)
    wkt: str    # Projection for the shapefile
    field_types: dict    # {name: ogr.FieldDefn, ...}
    shape_col: str     # Name of shape column
    shape_type: int     # ogr.wkb* constants; try ogr.GeometryTypeToName(lyr_def.GetGeomType())

def read_df(fname, shape_col='shape'):
    """Reads a dataframe from a shapefile (or similar type of vector file)

    fname:
        Name of file to read.
        (Can be used to read from zip file)
    shape_col:
        Name to call the column containing the actual ogr.Geometry
        (None if you only want to read the metadata)
    """
    ds = ogr.Open(fname)
    layer = ds.GetLayer(0)

    srs = layer.GetSpatialRef()
    wkt = srs.ExportToWkt()

    # Identify the attribute names and field definitions (includes type)
    # https://gdal.org/doxygen/classOGRFieldDefn.html
    defn = layer.GetLayerDefn()
    field_types = dict()
    for i in range(defn.GetFieldCount()):
        name = defn.GetFieldDefn(i).GetName()
        field_types[name] = defn.GetFieldDefn(i).GetType()

    shape_type = layer.GetGeomType() if shape_col is not None else None

    # Read the data
    rows = list()
    for feature in layer:
        row = [feature.GetField(i) for i in range(len(field_types))]
        if shape_col is not None:
            geometry_ref = feature.GetGeometryRef()
#            print('type geometry_ref ', type(geometry_ref))
            row.append(geometry_ref.Clone())
        rows.append(row)

    df = pd.DataFrame(rows, columns=list(field_types.keys()) + [shape_col])

    return Shapefile(df, wkt, field_types, shape_col, shape_type)
# -----------------------------------------------------
def polygon(coords):
    """Creates an OGR Polygon
    coords: [(x,y), ...]"""
    # https://pcjericks.github.io/py-gdalogr-cookbook/geometry.html#create-a-polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for point in coords:
        ring.AddPoint(*point)
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly

# -----------------------------------------------------
dtype2ogr = {
    np.dtype('int64'): ogr.OFTInteger64,
    np.dtype('int32'): ogr.OFTInteger,
    np.dtype('float64'):  ogr.OFTReal,
    np.dtype('float32'):  ogr.OFTReal,
    pd.StringDtype():   ogr.OFTString,
}

def field_types(df, field_types=dict()):
    """Determine type of each field from columns of dataframe.
    Returns: {name: ogr.FieldDefn, ...}
    """

    # Build a new field_types dict, in case we missed some
    if field_types is None:
        field_types = {}    # Dummy lookup

    fts = list()
    for cname in df.columns:
        if cname in field_types:
            # Use field type we already have
            fts.append((cname, field_types[cname]))
        else:
            dtype = df[cname].dtype
            try:
                ogrtype = dtype2ogr[dtype]
            except KeyError:
                print(f'Error on column {cname}')
                raise
            fts.append((cname, ogrtype))
    return dict(fts)
# -----------------------------------------------------
def write_df(sf, ofname):
    """sf: Shapefile
        Same type as output of read_df()
    """

    # Split the shape column into a separate dataframe
    shape_series = sf.df[[sf.shape_col]]
    df1 = sf.df.drop(sf.shape_col, axis=1)

    # Determine field definitions (fill in any missing definitions from dataframe)
    xfield_types = field_types(df1, sf.field_types)
#    field_types = sf.field_types if sf.field_types is not None else field_types(sf.df)

    # Open the shapefile
    ds = ogr.GetDriverByName('Esri Shapefile').CreateDataSource(str(ofname))
    layer = ds.CreateLayer('', osrutil.wkt_to_srs(sf.wkt), sf.shape_type)

    # Add attributes
    for name,ftype in xfield_types.items():
        print('xxx ', name, ftype)
        layer.CreateField(ogr.FieldDefn(name, ftype))

    # --------------------------------
    defn = layer.GetLayerDefn()
    for (_,shaperow), (_,row) in zip(shape_series.iterrows(), df1.iterrows()):
        feat = ogr.Feature(defn)
        feat.SetGeometry(shaperow[sf.shape_col])
        for field,value in zip(xfield_types.keys(), row):
#        for field,value in fields.items():
            feat.SetField(field, value)
        layer.CreateFeature(feat)

    # https://gdal.org/doxygen/classOGRGeometry.html
    # --------------------------------


    # Free memory with OGR
#    layer = None
#    ds = None

# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------
#https://pcjericks.github.io/py-gdalogr-cookbook/vector_layers.html#create-a-new-shapefile-and-add-data
