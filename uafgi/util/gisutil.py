import numpy as np
from osgeo import osr

# Simple Cartesian CRS ("Ortographic")
def _ortho_wkt():
    p = osr.SpatialReference()
    p.ImportFromProj4("+proj=ortho+lat_0=0+lon_0=0+x_0=0+y_0=0")
    return p.ExportToWkt()
ortho_wkt = _ortho_wkt()

#def invert_geotransform(GT):
#    # The GeoTransform is a tuple of 6 values,
#    # which relate raster indices into
#    # coordinates.
#    # Xgeo = GT[0] + Xpixel*GT[1] + Yline*GT[2]
#    # Ygeo = GT[3] + Xpixel*GT[4] + Yline*GT[5]
#    # https://www.mathsisfun.com/algebra/matrix-inverse.html
#
#    a = GT[1]
#    b = GT[2]
#    c = GT[4]
#    d = GT[5]
#    det = a * d - b * c
#    bydet = 1./det
#    # Inverse matrix: (a,b,c,d)
#    return (d*bydet, -b*bydet, -c*bydet, a*bydet)
# -------------------------------------------------------

# Taken from original GDAL Source
# https://github.com/OSGeo/gdal/blob/master/alg/gdaltransformer.cpp
def invert_geotransform(gt_in):
    gt_out = np.zeros(6)

    # Special case - no rotation - to avoid computing determinate
    # and potential precision issues.
    if (gt_in[2] == 0.0) and (gt_in[4] == 0.0) and \
        (gt_in[1] != 0.0) and (gt_in[5] != 0.0):

        #  X = gt_in[0] + x * gt_in[1]
        #  Y = gt_in[3] + y * gt_in[5]
        #  -->
        #  x = -gt_in[0] / gt_in[1] + (1 / gt_in[1]) * X
        #  y = -gt_in[3] / gt_in[5] + (1 / gt_in[5]) * Y
        gt_out[0] = -gt_in[0] / gt_in[1]
        gt_out[1] = 1.0 / gt_in[1]
        gt_out[2] = 0.0
        gt_out[3] = -gt_in[3] / gt_in[5]
        gt_out[4] = 0.0
        gt_out[5] = 1.0 / gt_in[5]

    else:
        # Assume a 3rd row that is [1 0 0].

        # Compute determinate.
        det = gt_in[1] * gt_in[5] - gt_in[2] * gt_in[4]
        magnitude = max(
                max(abs(gt_in[1]), abs(gt_in[2])),
                max(abs(gt_in[4]), abs(gt_in[5])))

        # Avoid divide by zero
        if abs(det) > 1e-10 * magnitude * magnitude:
            inv_det = 1.0 / det

            # Compute adjoint, and divide by determinate.

            gt_out[1] =  gt_in[5] * inv_det
            gt_out[4] = -gt_in[4] * inv_det

            gt_out[2] = -gt_in[2] * inv_det
            gt_out[5] =  gt_in[1] * inv_det

            gt_out[0] = ( gt_in[2] * gt_in[3] - gt_in[0] * gt_in[5]) * inv_det
            gt_out[3] = (-gt_in[1] * gt_in[3] + gt_in[0] * gt_in[4]) * inv_det

    return gt_out


# -------------------------------------------------------------------------------
class RasterInfo:
    """Reads spatial extents from GDAL NetCDF raster file.
    Currently only works for NetCDF raster files.

    May be used, eg, as:
                '-projwin', str(x0), str(y1), str(x1), str(y0),
                '-tr', str(dx), str(dy),
    Returns:
        self.x0, self.x1:
            Min, max of region in the file
        self.dx:
            Grid spacing in x direction
        geotransform: np.array(6)
            GDAL domain used in this file

    A geotransform consists in a set of 6 coefficients:
        GT(0) x-coordinate of the upper-left corner of the upper-left pixel.
        GT(1) w-e pixel resolution / pixel width.
        GT(2) row rotation (typically zero).
        GT(3) y-coordinate of the upper-left corner of the upper-left pixel.
        GT(4) column rotation (typically zero).
        GT(5) n-s pixel resolution / pixel height (negative value for a north-up image, i.e. the standard for GeoTIFF where the northernmost row of pixels comes first in the file).
    """

    def __init__(self, wkt, nx, ny, geotransform):
        self.wkt = wkt
        self.nx = nx
        self.ny = ny
        self.geotransform = geotransform

        # Calculated items
        GT = self.geotransform
        self.geoinv = invert_geotransform(GT)
        self.x0 = GT[0]
        self.y0 = GT[3]
        self.dx = GT[1]
        self.dy = GT[5]

        self.x1 = self.x0 + self.nx * self.dx
        self.y1 = self.y0 + self.ny * self.dy

    @property
    def nxy(self):
        return self.nx * self.ny

    @property
    def shape(self):
        return (self.ny, self.nx)

    def flipud(self):
        """Turns a north-up geometry into north-down (or vice versa)"""
        GT = self.geotransform
        return RasterInfo(self.wkt, self.nx, self.ny,
            np.array([
                GT[0], GT[1], GT[2],
                GT[3]+GT[5]*self.ny, GT[4], -GT[5]]))


    def __str__(self):
        return f"""RasterInfo:
    x: [{self.x0}, {self.x1}] dx={self.dx} nx={self.nx}
    y: [{self.y0},{self.y1}] dy={self.dy} ny={self.ny}
    geotransform: {self.geotransform}
    wkt: {self.wkt}"""

    def __repr__(self):
        return self.__str__()

    @property
    def centersx(self):
        halfx = 0.5 * self.dx
        return np.arange(self.x0 + halfx, self.x1 + halfx, self.dx)

    @property
    def centersy(self):
        halfy = 0.5 * self.dy
        return np.arange(self.y0 + halfy, self.y1 + halfy, self.dy)

    # -----------------------------------------------------

    @property
    def extents(self):
        """Provide extents for Cartopy / Matplotlib's ax.set_extent()"""
        gt = self.geotransform
        x0 = gt[0]
        x1 = x0 + gt[1] * self.nx

        y0 = gt[3]
        y1 = y0 + gt[5] * self.ny

        return [x0,x1,y0,y1]


    def to_xy(self, i, j):
        """Converts an (i,j) pixel address to an (x,y) geographic value"""
        GT = self.geotransform
        Xgeo = GT[0] + i*GT[1] + j*GT[2]
        Ygeo = GT[3] + i*GT[4] + j*GT[5]
        return Xgeo,Ygeo


    def to_ij(self, x, y):
        """Converts an (x,y) value to an (i,j) index into raster
        NOTE: Similar to geoio.GeoImage.proj_to_raster()"""

        # https://stackoverflow.com/questions/40464969/why-does-gdal-grid-turn-image-upside-down
        GT = self.geoinv
        ir = GT[0] + x*GT[1] + y*GT[2]
        jr = GT[3] + x*GT[4] + y*GT[5]

#        return np.round(ir), np.round(jr)
        return np.round(ir).astype('i'), np.round(jr).astype('i')

# ====================================================
# From PISM
# https://github.com/pism/pism/blob/main/util/fill_missing_petsc.py
