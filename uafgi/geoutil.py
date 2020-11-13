#from osgeo import ogr
import shapefile
import shapely.geometry
import pyproj
import numpy as np

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




def read_shapes(fname):
    """Straight shapefile reader"""
    with shapefile.Reader(fname) as sf:
        field_names = [x[0] for x in sf.fields[1:]]
        for i in range(0, len(sf)):
            shape = sf.shape(i)
            rec = sf.record(i)
            attrs = dict(zip(field_names, rec))

            yield (shape,attrs)
