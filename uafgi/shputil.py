import os
import numpy as np
import pyproj
import subprocess
import pathlib
import pandas as pd

from osgeo import ogr,gdal
import shapefile
import shapely.geometry

from uafgi import gdalutil,osrutil,pdutil

shapely2ogr = {
    'Polygon' : ogr.wkbPolygon,
    'MultiPolygon' : ogr.wkbMultiPolygon,
}
    
def write_shapefile(shapely_obj, fname):
    """Writes a single Shapely object into a shapefile"""

    ogr_type = shapely2ogr[shapely_obj.geom_type]

    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(fname)
    layer = ds.CreateLayer('', None, ogr_type)

    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    defn = layer.GetLayerDefn()

    ## If there are multiple geometries, put the "for" loop here

    # Create a new feature (attribute and geometry)
    feat = ogr.Feature(defn)
    feat.SetField('id', 123)

    # Make a geometry, from Shapely object
    geom = ogr.CreateGeometryFromWkb(shapely_obj.wkb)
    feat.SetGeometry(geom)

    layer.CreateFeature(feat)

    # ------- Local variables are all destroyed
    # feat = geom = None  # destroy these
    # Save and close everything
    # ds = layer = feat = geom = None

def write_shapefile2(shapely_objs, fname, fields=[], attrss=[]):
    """Writes a single Shapely object (or list of Shaeply objects) into a shapefile
    fields: [ogr.FieldDefn]
        List of field definitions to set up in this file.
        Eg: [ogr.FieldDefn('id', ogr.OFTInteger), ...]
    attrs: [(...), (...), ...]
        List of dicts: attributes to store in the fields
    """

    if len(shapely_objs) == 0:
        print('Nothing to write in shapefile: {}'.format(fname))
        try:
            os.remove(fname)
        except OSError:
            pass
        return

    # Get canonical instance; assume all objects are the same
    obj0 = shapely_objs[0]

    ogr_type = shapely2ogr[obj0.geom_type]

    # Now convert it to a shapefile with OGR    
    driver = ogr.GetDriverByName('Esri Shapefile')
    ds = driver.CreateDataSource(fname)
    layer = ds.CreateLayer('', None, ogr_type)
    defn = layer.GetLayerDefn()

    # Add one attribute
    for field in fields:
        layer.CreateField(field)
#        field_names.append(field_def.GetName())


    ## If there are multiple geometries, put the "for" loop here
    for obj,attrs in zip(shapely_objs,attrss):

        # Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        for ix,(field,attr) in enumerate(zip(fields,attrs)):
            name = field.GetName()
            feat.SetField(ix, attr)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(obj.wkb)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)

    # ------- Local variables are all destroyed
    # feat = geom = None  # destroy these
    # Save and close everything
    # ds = layer = feat = geom = None




def _xPOLYGON(shape,transform_fn):
    gline_xx,gline_yy = transform_fn(
        np.array([xy[0] for xy in shape.points]),
        np.array([xy[1] for xy in shape.points]))
    return None,shapely.geometry.Polygon(zip(gline_xx, gline_yy))

def _xPOINT(shape,transform_fn):
    pt = shape.points[0]
    xx,yy = transform_fn(pt[0], pt[1])
    return shapely.geometry.Point(pt[0],pt[1]), shapely.geometry.Point(xx,yy)

def _xPOLYLINE(shape,transform_fn):
    gline_xx,gline_yy = transform_fn(
        np.array([xy[0] for xy in shape.points]),
        np.array([xy[1] for xy in shape.points]))
    return None,shapely.geometry.LineString(zip(gline_xx, gline_yy))


# https://github.com/GeospatialPython/pyshp/blob/master/shapefile.py
# Constants for shape types
# NULL = 0
# POINT = 1
# POLYLINE = 3
# POLYGON = 5
# MULTIPOINT = 8
# POINTZ = 11
# POLYLINEZ = 13
# POLYGONZ = 15
# MULTIPOINTZ = 18
# POINTM = 21
# POLYLINEM = 23
# POLYGONM = 25
# MULTIPOINTM = 28
# MULTIPATCH = 31

shapely_converters = {
    shapefile.POLYGON : _xPOLYGON,
    shapefile.POLYLINE : _xPOLYLINE,
    shapefile.POINT : _xPOINT,
    }

def get_transformer(fname, wkt1):
    """Creates a transformer from native projetion to wkt1"""

    # Convert WKT to CRS
    crs1 = pyproj.CRS.from_string(wkt1)

    # Get CRS out of shapefile
    with open(fname[:-4] + '.prj') as fin:
        crs0 = pyproj.CRS.from_string(next(fin))

    # Converts from crs0 to crs1
    # See for always_xy: https://proj.org/faq.html#why-is-the-axis-ordering-in-proj-not-consistent
    return pyproj.Transformer.from_crs(crs0, crs1, always_xy=True)

def read(fname, read_shapes=True, wkt=None):
    """read_shapes:
        Should the shape actually be read?  (Or just the attributes)?
    """

    if read_shapes:
        proj = get_transformer(fname, wkt)

    with shapefile.Reader(fname) as reader:
        #fields = reader.fields
        for i in range(0, len(reader)):
            rec = reader.record(i).as_dict()

            if read_shapes:
                shape_raw = reader.shape(i)
                # A Shapefile end up with "holes" if shapes have been removed from it.
                if shape_raw.shapeType == shapefile.NULL:
                    continue

#                print('i=',i)
#                print('shape_raw.shapeType = {}'.format(shape_raw.shapeType))
#                print('shape_raw = {}'.format(shape_raw))

                shape0,shape = shapely_converters[shape_raw.shapeType](shape_raw, proj.transform)
                rec['_shape0'] = shape0    # Raw coordinates
                rec['_shape'] = shape
            yield rec


def read_df(fname, read_shapes=True, wkt=None, shape0=None, shape='loc', add_prefix=None):
    """
    wkt:
        Project shapes into this projection(if they are being read).
    read_shapes:
        Should the acutal shapes be read?  Or just the metadata?
    shape0:
        Name to call the "shape0" columns when all is said and done
        (i.e. the original shape, before it was reprojected)
    Returns columns:
        fid:
            File ID, the ID used to read this record back with a ShapeFile reader
    """

    df = pd.DataFrame(read(fname, wkt=wkt, read_shapes=read_shapes))
    df = df.reset_index().rename(columns={'index':'fid'})    # Add a key column

    drops = list()
    renames = dict()

    # ------- Rename or drop shape-related columns
    if read_shapes:
        if shape0 is None:
            drops.append('_shape0')
        else:
            renames['_shape0'] = shape0

        if shape is None:
            drops.append('_shape')
        else:
            renames['_shape'] = shape

    df = df.drop(drops, axis=1).rename(columns=renames)
    up = pdutil.ext_df(df, wkt, add_prefix=add_prefix,
        units={},
        keycols=['fid'])

    return up

# ---------------------------------------------------------
# Here's an example of reading a shapefile using ogr
# def read_fjords(dest_crs_wkt):
#     """
#     dest_crs_wkt:
#         WKT of the desination coordinate system.
#     """
# 
#     driver = ogr.GetDriverByName('ESRI Shapefile')
#     src_ds = driver.Open('troughs/shp/fjord_outlines.shp')
#     src_lyr = src_ds.GetLayer()   # Put layer number or name in her
#     src_srs = src_lyr.GetSpatialRef()
#     dst_srs = osr.SpatialReference()
#     dst_srs.ImportFromWkt(dest_crs_wkt)
#     transform = osr.CoordinateTransformation(src_srs, dst_srs)
# 
#     fjords_s = list()
#     if True:
#         while True:
#             feat = src_lyr.GetNextFeature()
#             if feat is None:
#                 break
#             poly = ogrutil.to_shapely_polygon(feat,transform)
# 
#             fjords_s.append(poly)
# 
#     return fjords_s
##    fjords = pd.Series(name='fjords',data=fjords_s)
##    return fjords
# ---------------------------------------------------------

#class ShapefileReader(object):
#    """Shapefile reader, augmented to convert to desired projection."""
#
#    def __init__(self, fname, crs1):
#        self.fname = fname
#        self.crs1 = crs1    # Projection to translate to
#
#    def __enter__(self):
#        self.reader = shapefile.Reader(self.fname)
#        self.fields = self.reader.fields
#
#        # Get CRS out of shapefile
#        with open(self.fname[:-4] + '.prj') as fin:
#            self.crs0 = pyproj.CRS.from_string(next(fin))
#
#        # Converts from self.crs0 to self.crs1
#        # See for always_xy: https://proj.org/faq.html#why-is-the-axis-ordering-in-proj-not-consistent
#        self.proj = pyproj.Transformer.from_crs(self.crs0, self.crs1, always_xy=True)
#        return self
#
#    def __exit__(self, exc_type, exc_value, exc_traceback):
#        self.reader.__exit__(exc_type, exc_value, exc_traceback)
#
#    def __len__(self):
#        return len(self.reader)
#
#    def shape(self, ix):
#        """Read a shape, reproject and convert to Polygon"""
#        shape = self.reader.shape(ix)
#        if shape.shapeType != shapefile.POLYGON:
#            raise ValueError('shapefile.POLYGON shapeType expected in file {}'.format(self.fname))
#
#        gline_xx,gline_yy = self.proj.transform(
#            np.array([xy[0] for xy in shape.points]),
#            np.array([xy[1] for xy in shape.points]))
#        return shapely.geometry.Polygon(zip(gline_xx, gline_yy))
#
#    def records(self):
#        for i in len(self.reader):
#            rec = sf.record(i).as_dict()
#            poly = self.polygon(i)
#            yield rec,poly



class ShapefileWriter(object):
    """Writes Shapely objects into a shapefile"""

    def __init__(self, fname, shapely_type, field_defs, wkt=None):
        """
        fname:
            Name of file to create
        shapely_type: str
            Type of Shapely object that will be written here
            Eg: 'Polygon', 'MultiPolygon'
        field_defs: ((name,type), ...)
            name: Name of attribute field
            type: ogr.OFTInteger, ogr.OFTString, etc.
                  https://gdal.org/java/org/gdal/ogr/ogrConstants.html
        wkt: str
            WKT of the projection to use for this Shapefile
        """
        self.fname = fname
        self.shapely_type = shapely_type
        self.field_defs = field_defs
        self.wkt = wkt

    def __enter__(self):
        ogr_type = shapely2ogr[self.shapely_type]

        # Now convert it to a shapefile with OGR    
        self.driver = ogr.GetDriverByName('Esri Shapefile')
        self.ds = self.driver.CreateDataSource(self.fname)
        self.layer = self.ds.CreateLayer('', osrutil.wkt_to_srs(self.wkt), ogr_type)

        # Add attributes
#        print('fd ',self.field_defs)
        for name,ftype in self.field_defs:
            self.layer.CreateField(ogr.FieldDefn(name, ftype))
        #self.layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.layer = None

    def write(self, shapely_obj, **fields):
#        if shapely_obj.geom_type != self.shapely_type:
#            raise TypeError('Trying to write an object of type {} to a file of type {}'.format(shapely_obj.geom_type, self.shapely_type))

#        ogr_type = shapely2ogr[shapely_obj.geom_type]

        ## If there are multiple geometries, put the "for" loop here

        # Create a new feature (attribute and geometry)
        defn = self.layer.GetLayerDefn()
        feat = ogr.Feature(defn)
        for field,value in fields.items():
            feat.SetField(field, value)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkb(shapely_obj.wkb)
        feat.SetGeometry(geom)

        self.layer.CreateFeature(feat)

        # ------- Local variables are all destroyed
        # feat = geom = None  # destroy these
        # Save and close everything
        # ds = self.layer = feat = geom = None


def get_crs(shapefile):
    """Reads the coordinate reference system (CRS) of a shapefile.

    shapefile: *.shp
        The .shp fork of a shapefile
    Returns:
        String of CRS
    """

    with open(shapefile[:-4] + '.prj') as fin:
        wks_s = next(fin)
    termini_crs = pyproj.CRS.from_string(wks_s)


def select_feature(ishapefile, fid, oshapefile):
    """Selects a single feature out of a shapefile and stores into a new shapefile
    ishapefile:
        Input shapefile
    fid:
        ID of feature to select (0-based)
    oshapefile:
        Output shapefile
    """

    # Select a single polygon out of the shapefile
    cmd = ['ogr2ogr', oshapefile, ishapefile, '-fid', str(fid)]
    #print(' '.join(cmd))
    subprocess.run(cmd, check=True)

def fjord_mask(termini_closed_file, index, geometry_file, tdir):
    """Converts a closed polygon in a shapefile into

    termini_closed_file:
        Shapefile containing the closed terminus polygons.
        One side of the polygon is the terminus; the rest is nearby parts of the fjord.
    index:
        Which polygon (stargin with 0) in the terminus shapefile to use.
    tdir: ioutil.TmpDir
        Location for temporary files
    """

    with ioutil.tmp_dir(odir, tdir='tdir') as tdir:

        one_terminus = os.path.join(tdir, 'one_terminus.shp')
        select_feature(termini_closed_file, index, one_terminus)

        # Cut the bedmachine file based on the shape
        cut_geometry_file = os.path.join(tdir, 'cut_geometry_file.nc')
        cmd = ['gdalwarp', '-cutline', one_terminus, 'NETCDF:{}:bed'.format(geometry_file), cut_geometry_file]
        subprocess.run(cmd, check=True)

        # Read the fjord mask from that file
        with netCDF4.Dataset(cut_geometry_file) as nc:
            fjord = nc.variables['Band1'][:].mask

    return fjord


def check_error(err):
    if err != 0:
        raise 'GDAL Error {}'.format(err)

def crs(shapefile):
    """Reads the coordinate reference system (CRS) out of a shapefile.
    (Actually, out of a shapefile's .prj file)

    shapefile:
        Name of the shapefile (with or without .shp extension)
    Returns:
        CRS as a string
    """
    fname = os.path.splitext(shapefile)[0] + '.prj'
    with open(fname) as fin:
        crs = pyproj.CRS.from_string(next(fin))
    return crs
