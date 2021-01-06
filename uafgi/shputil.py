from osgeo import ogr
import shapefile
import shapely.geometry
import pyproj
import numpy as np

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



class ShapefileReader(object):
    """Shapefile reader, augmented to convert to desired projection."""

    def __init__(self, fname, crs1):
        self.fname = fname
        self.crs1 = crs1    # Projection to translate to

    def __enter__(self):
        self.reader = shapefile.Reader(self.fname)

        # Get CRS out of shapefile
        with open(self.fname[:-4] + '.prj') as fin:
            self.crs0 = pyproj.CRS.from_string(next(fin))

        # Converts from self.crs0 to self.crs1
        # See for always_xy: https://proj.org/faq.html#why-is-the-axis-ordering-in-proj-not-consistent
        self.proj = pyproj.Transformer.from_crs(self.crs0, self.crs1, always_xy=True)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.reader.__exit__(exc_type, exc_value, exc_traceback)

    def polygon(self, ix):
        """Read a shape, reproject and convert to Polygon"""
        shape = self.reader.shape(ix)
        if shape.shapeType != shapefile.POLYGON:
            raise ValueError('shapefile.POLYGON shapeType expected in file {}'.format(self.fname))

        gline_xx,gline_yy = self.proj.transform(
            np.array([xy[0] for xy in shape.points]),
            np.array([xy[1] for xy in shape.points]))
        return shapely.geometry.Polygon(zip(gline_xx, gline_yy))


class ShapefileWriter(object):
    """Writes Shapely objects into a shapefile"""

    def __init__(self, fname, shapely_type, field_defs):
        """
        fname:
            Name of file to create
        shapely_type: str
            Type of Shapely object that will be written here
            Eg: 'Polygon', 'MultiPolygon'
        field_defs: ((name,type), ...)
            name: Name of attribute field
            type: ogr.OFTInteger, etc.
                  https://gdal.org/java/org/gdal/ogr/ogrConstants.html
        """
        self.fname = fname
        self.field_defs = field_defs
        self.shapely_type = shapely_type

    def __enter__(self):
        ogr_type = shapely2ogr[self.shapely_type]

        # Now convert it to a shapefile with OGR    
        self.driver = ogr.GetDriverByName('Esri Shapefile')
        self.ds = self.driver.CreateDataSource(self.fname)
        self.layer = self.ds.CreateLayer('', None, ogr_type)

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
