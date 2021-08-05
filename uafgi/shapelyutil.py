import itertools
from osgeo import ogr,osr
import shapely.geometry


# Convert Shapely type to OGR type
shapely_to_ogr_type = {
    shapely.geometry.linestring.LineString: ogr.wkbLineString,
    shapely.geometry.polygon.Polygon: ogr.wkbPolygon,
}


def to_datasource(shape):
    """Converts an in-memory Shapely object to an (open) OGR datasource.
    shape:
        A shapely polygon (or LineString, etc)
    """

    # Create an OGR in-memory datasource with our single polygon
    ds=ogr.GetDriverByName('MEMORY').CreateDataSource('memData')
    layer = ds.CreateLayer('', None, shapely_to_ogr_type[type(shape)])#ogr.wkbPolygon)
    feat = ogr.Feature(layer.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromWkb(shape.wkb))
    layer.CreateFeature(feat)

    return ds

def pointify(shapes):
    """Converts a bunch of shapes into a jumbled list of points in the
    shapes combined.  Shapes could be LineString, Polygon, etc.

    shapes:
        Iterable of Shapely shapes
    """
    return shapely.geometry.MultiPoint(list(itertools.chain(*(shape.coords for shape in shapes))))
