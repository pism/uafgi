from osgeo import ogr,osr

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
