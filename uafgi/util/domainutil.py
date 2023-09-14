from uafgi.
class DomainGrid(gisutil.RasterInfo):    # (gridD)
    """Define a bunch of rectangles indexed by (idom, jdom).

    NOTE: This is a subclass of gisutil.RasterInfo.  Each "gridcell"
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
        y0 = yy[0]
        y1 = yy[2]

        # The domain grid should have the same north-up / north-down
        # as the original grid it's on top of.
        assert x0 < x1
        assert y0 < y1

        dx = domain_size[0] #* xsgn
        dy = domain_size[1] #* ysgn

        # Round region to integral domain size
        x0 = dx * math.floor(x0/dx)
        y0 = dy * math.floor(y0/dy)

        #xsgn = np.sign(x1-x0)
        #ysgn = np.sign(y1-y0)

        # Number of domains in the overall experiment region
        nx = math.ceil((x1-x0)/dx)    # type=int
        ny = math.ceil((y1-y0)/dy)

        # Geotransform
        GT = np.array([x0, dx, 0, y0, 0, dy])
        super().__init__(wkt, nx, ny, GT)

        self.domain_margin = domain_margin
        self.index_box = index_box

    def sub(self, i, j, dx, dy, margin=True):
        """Produes a sub-grid for the (i,j) domain.
        dx,dy:
            Resolution of the subgrid
        margin:
            Should a margin be kept around the subgrid?
        """

        if margin:
            mx = np.sign(self.dx) * self.domain_margin[0]
            my = np.sign(self.dy) * self.domain_margin[1]
        else:
            mx = 0
            my = 0


        GT = self.geotransform
        x0 = GT[0] + GT[1] * i - mx
        y0 = GT[3] + GT[5] * j - my

        nx = (self.dx + 2*mx)/ dx
        ny = (self.dy + 2*my)/ dy

        return gisutil.RasterInfo(self.wkt, nx, ny, np.array([x0, dx, 0, y0, 0, dy]))


    def global(self, dx, dy):
        """Produces a grid for the ENTIRE extent of gridD, at (dx,dy) fine resolution.

        NOTES:
           * The origin of this grid is the same as origin of self.subgrid(0,0)
        """

        GT = self.self.geotransform
        x0 = GT[0]
        y0 = GT[3]
        nx = self.nx * (self.dx / dx)
        ny = self.ny * (self.dy / dy)

        return gisutil.RasterInfo(self.wkt, nx, ny, np.array([x0, dx, 0, y0, 0, dy]))

