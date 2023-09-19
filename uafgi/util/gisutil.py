import functools,math
import numpy as np
from osgeo import osr
import shapely
from uafgi.util import shputil

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
        self.nx = int(nx)
        self.ny = int(ny)
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


    def to_xy(self, i, j, center=False):
        """Converts an (i,j) pixel address to an (x,y) geographic value
        center:
            Convert to center or corner of gridcell?"""

        GT = self.geotransform
        Xgeo = GT[0] + i*GT[1] + j*GT[2] + (0.5*self.dx if center else 0)
        Ygeo = GT[3] + i*GT[4] + j*GT[5] + (0.5*self.dy if center else 0)
        return Xgeo,Ygeo


    def to_ij(self, x, y):
        """Converts an (x,y) value to an (i,j) index into raster
        NOTE: Similar to geoio.GeoImage.proj_to_raster()"""

        # https://stackoverflow.com/questions/40464969/why-does-gdal-grid-turn-image-upside-down
        GT = self.geoinv
        ir = GT[0] + x*GT[1] + y*GT[2]
        jr = GT[3] + x*GT[4] + y*GT[5]

#        return np.round(ir), np.round(jr)
#        return np.round(ir).astype('i'), np.round(jr).astype('i')

        # if x0=0 and dx=10, then anything x \in [0,10) should have i=0
        return np.floor(ir).astype('i'), np.floor(jr).astype('i')


    @property
    def bounding_box(self):
        """Returns: Shapely rectangle of the bounding box of this grid."""

        GT = self.geotransform
        x0 = self.x0
        x1 = self.x0 + self.nx * self.dx
        y0 = self.y0
        y1 = self.y0 + self.ny * self.dy

        coords = [
            (x0,y0),
            (x1,y0),
            (x1,y1),
            (x0,y1),
            (x0,y0)]
        return shapely.geometry.Polygon(coords)


class DomainGrid(RasterInfo):    # (gridD)
    """Define a bunch of rectangles indexed by (idom, jdom).
    (0,0) is in the north-west of the region.  ("North-up" order, consistent with typical GeoTIFF)

    NOTE: This is a subclass of RasterInfo.  Each "gridcell"
          in RasterInfo represents a domain in DomainGrid.
    """

    def __init__(self, wkt, index_region_shp, domain_size, domain_margin):

        """
        wkt:
            Projection to use.
        index_region_shp: shapefile name
            A simple polygon of the ENTIRE region that MIGHT be
            covered.  (i.e. all of Alaska).  This region is divided up
            into domain-size rectangles, which are given (idom,jdom)
            index numbers.  All portions of the experiment must happen
            INSIDE this region.
        domain_size: (x,y)
            Size of each domain
        domain_margin: (x,y)
            Amount of margin to add to each domain.
            (For avalanches that start near the edge and run out).
        """

        # Load the overall index region
        index_region = list(shputil.read(index_region_shp))[0]['_shape']
#        print('index_region ', index_region)
        index_box = index_region.envelope  # Smallest rectangle with sides oriented to axes
#        print('index_box ', index_box)

        # ----------------------------------------
        # Determine "domain geotransform" based on index_box
        domain_size = domain_size
        domain_margin = domain_margin
        xx,yy = index_box.exterior.coords.xy
        x0 = xx[0]
        x1 = xx[1]
        y0 = yy[2]    # Enforce North-up order
        y1 = yy[0]

        xsgn = np.sign(x1-x0)
        ysgn = np.sign(y1-y0)

        # The domain grid should have the same north-up / north-down
        # as the original grid it's on top of.
        assert x0 < x1
        assert y0 > y1    # North-up order

        dx = domain_size[0] * xsgn
        dy = domain_size[1] * ysgn

        # Round region to integral domain size
        x0 = dx * math.floor(x0/dx)
        y0 = dy * math.ceil(y0/dy)    # North-up order


        # Number of domains in the overall experiment region
        nx = math.ceil((x1-x0)/dx)    # type=int
        ny = math.ceil((y1-y0)/dy)

        # Geotransform
        GT = np.array([x0, dx, 0, y0, 0, dy])
        super().__init__(wkt, nx, ny, GT)

        self.domain_margin = domain_margin
        self.index_box = index_box

    def poly(self, ix, iy, margin=False):
        """Returns given rectangle by index.
        ix, iy:
            Coordinates of the domain within the overall region.
        margin:
            Should the margin be included in the box returned?
        Returns:
            A rectangular polygon, oriented in standard (counter clockwise) fashion.
        """

        GT = self.geotransform
        x0 = GT[0] + GT[1] * ix
        y0 = GT[3] + GT[5] * iy

        if margin:
            mx = np.sign(self.dx) * self.domain_margin[0]
            my = np.sign(self.dy) * self.domain_margin[1]
        else:
            mx = 0
            my = 0

        coords = [
            (x0-mx, y0-my),
            (x0+self.dx+mx, y0-my),
            (x0+self.dx+mx, y0+self.dy+my),
            (x0-mx, y0+self.dy+my),
            (x0-mx, y0-my)]
        return shapely.geometry.Polygon(coords)

    def subgrid(self, minx, miny, maxx, maxy, resx, resy):
        xsgn = np.sign(self.dx)
        ysgn = np.sign(self.dy)

        x0 = minx if xsgn>0 else maxx
        y0 = miny if ysgn>0 else maxy

        # Get sign of (dx,dy) same as (self.dx, self.dy)
        dx = abs(resx)*xsgn
        dy = abs(resy)*ysgn

        GT = self.geotransform

        nx = int(0.5 + (maxx - minx) / dx)
        ny = int(0.5 + (maxy - miny) / dy)

        grid = RasterInfo(self.wkt, nx, ny, np.array([x0, dx, 0, y0, 0, dy]))

        return grid


    def sub(self, i, j, resx, resy, margin=True):
        """Produes a sub-grid for the (i,j) domain (north-up)
        resx,resy:
            Resolution of the subgrid
            (Both positive numbers; will be adjusted based on self.dx / self.dy)
        margin:
            Should a margin be kept around the subgrid?
        Returns: RasterInfo with extra fields
            grid.gridG
                The global hi-res grid we are a logical part of
            grid.i0, grid.j0
                Origin of this grid within gridG
        """

        xsgn = np.sign(self.dx)
        ysgn = np.sign(self.dy)

        # Get sign of (dx,dy) same as (self.dx, self.dy)
        dx = abs(resx)*xsgn
        dy = abs(resy)*ysgn

        if margin:
            mx = xsgn * self.domain_margin[0]
            my = ysgn * self.domain_margin[1]
        else:
            mx = 0
            my = 0


        GT = self.geotransform

        offsetx = self.dx * i - mx
        x0 = self.x0 + offsetx

        offsety = self.dy * j - my
        y0 = self.y0 + offsety

        nx = int((self.dx + 2*mx)/ dx + 0.5)
        ny = int((self.dy + 2*my)/ dy + 0.5)

        grid = RasterInfo(self.wkt, nx, ny, np.array([x0, dx, 0, y0, 0, dy]))

        # Add (i0, j0) indicating this is a subset of the global grid
        grid.gridG = self.global_grid(dx, dy)
        grid.i0 = int(offsetx / dx + 0.5)
        grid.j0 = int(offsety / dy + 0.5)

        return grid

    @functools.lru_cache()
    def global_grid(self, dx, dy):
        """Produces a grid for the ENTIRE extent of gridD, at (dx,dy) fine resolution.

        NOTES:
           * The origin of this grid is the same as origin of self.subgrid(0,0)
        """

        GT = self.geotransform
        x0 = GT[0]
        y0 = GT[3]
        nx = self.nx * (self.dx / dx)
        ny = self.ny * (self.dy / dy)

        return RasterInfo(self.wkt, nx, ny, np.array([x0, dx, 0, y0, 0, dy]))



# ====================================================
# From PISM
# https://github.com/pism/pism/blob/main/util/fill_missing_petsc.py
