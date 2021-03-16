from osgeo import osr

wkt_to_srs(wkt)
    """Creates an OSR (GDAL/OGR) SpatialReference object from a WKT projection string."""
    if wkt is None:
        return None
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return srs

