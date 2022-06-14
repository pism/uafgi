#from osgeo import ogr
import shapefile
import shapely.geometry
import pyproj
import numpy as np





def read_shapes(fname):
    """Straight shapefile reader (see shapelyutil.ShapefileReader)"""
    with shapefile.Reader(fname) as sf:
        field_names = [x[0] for x in sf.fields[1:]]
        for i in range(0, len(sf)):
            shape = sf.shape(i)
            rec = sf.record(i)
            attrs = dict(zip(field_names, rec))

            yield (shape,attrs)
