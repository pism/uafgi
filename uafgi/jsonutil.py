import json
from gdal import ogr,osr
from uafgi import cdoutil,ncutil,functional,gdalutil,ogrutil
from osgeo import gdal
import numpy as np

@functional.memoize
def _gdal_srs():
    """GeoJSON files are all EPSG 4326 (lon/lat coordinates).
    This is the same SRS (including axis mapping) that you will get if
    you ask OGR for the SRS:
        driver = ogr.GetDriverByName('GeoJSON')
        src_ds = driver.Open(polygon_file)
        src_lyr = src_ds.GetLayer()   # Put layer number or name in her
        inSpatialRef = src_lyr.GetSpatialRef()

    """

    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(4326)
    inSpatialRef.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    return inSpatialRef

GDAL_SRS = _gdal_srs()


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

