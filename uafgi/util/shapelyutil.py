import itertools
import numpy as np
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

# -----------------------------------------------------------
def _scale_vec(vec,margin):
    """Adds a certain length to a vector.  Helper function."""
    veclen = np.linalg.norm(vec)
    if (veclen+margin) < 0:
        raise ValueError('Margin is larger than side')
    factor = margin / veclen
    return factor*vec

def add_margin(p,margin):
    """Adds a margin to a (rotated) rectangle, i.e. a domain rectangle.
    p: shapely.geometry.Polygon
        The rectangle
    margin:
        Absolute amount to add to length and width.
        If negative, subtract this amount; cannot subtract more than original length
    """
    pts = np.array(p.boundary.coords)
    edges = np.diff(pts, axis=0)
    margin2 = .5*margin
    pts[0,:] += (_scale_vec(edges[3,:],margin2) - _scale_vec(edges[0,:],margin2))
    pts[1,:] += (_scale_vec(edges[0,:],margin2) - _scale_vec(edges[1,:],margin2))
    pts[2,:] += (_scale_vec(edges[1,:],margin2) - _scale_vec(edges[2,:],margin2))
    pts[3,:] += (_scale_vec(edges[2,:],margin2) - _scale_vec(edges[3,:],margin2))

    p = shapely.geometry.Polygon(list(zip(pts[:-1,0], pts[:-1,1])))
    return p
# -----------------------------------------------------------
