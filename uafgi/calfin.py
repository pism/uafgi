import pyproj
import datetime
import re
import os

def iterate_termini(termini_file, map_crs):
    """Iterate through a CALFIN terminus file (linestrings or closed polygons)
    yields: date, (gline_xx, gline_yy)
        date: datetime.date
            Date of terminus
        gline_xx, gline_yy: [],[]
            X and Y coordinates of the terminus
    """

    # The CRS associated with shapefile
    with open(termini_file[:-4] + '.prj') as fin:
        wks_s = next(fin)
    termini_crs = pyproj.CRS.from_string(wks_s)

    # Converts from termini_crs to map_crs
    # See for always_xy: https://proj.org/faq.html#why-is-the-axis-ordering-in-proj-not-consistent
    proj = pyproj.Transformer.from_crs(termini_crs, map_crs, always_xy=True)

    for shape,attrs in geoutil.read_shapes(termini_file):
        if shape.shapeType != shapefile.POLYGON:
            raise ValueError('shapefile.POLYGON shapeType expected in file {}'.format(termini_file))

        gline_xx,gline_yy = proj.transform(
            np.array([xy[0] for xy in shape.points]),
            np.array([xy[1] for xy in shape.points]))

        dt = datetime.datetime.strptime(attrs['Date'], '%Y-%m-%d').date()
        print(attrs)
        yield dt, (gline_xx, gline_yy)

class ParseFilename:
    calfinRE = re.compile(r'termini_(\d\d\d\d)-(\d\d\d\d)_([^_]+)_v(\d+)\.(\d+)\.(...)')
    def __init__(self, fname):
        """Parses a Calfin shapefile filename.
        Eg:
            termini_1972-2019_Bruckner-Gletscher_v1.0.shp
        """
        self.dir,leaf = os.path.split(fname)
        match = self.calfinRE.match(leaf)
        self.year0 = int(match.group(1))
        self.year1 = int(match.group(2))
        self.glacier_name = match.group(3)
        self.version0 = int(match.group(4))
        self.version1 = int(match.group(5))
        self.ext = match.group(6)

