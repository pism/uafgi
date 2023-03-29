import numpy as np
import netCDF4, cf_units
from uafgi.util import ogrutil,gisutil,shapelyutil,gdalutil
from osgeo import osr,ogr,gdal


def rasterize_polygons(polygon_ds, grid_info):
    """Rasterizes all polygons from polygon_ds into a single raster, which
    is returned as a Numpy array.

    polygon_ds:
        Open OGR dataset containing polygons in a single layer
        Can be Shapefile, GeoJSON, etc.
        Eg: poly_ds = gdalutil.open(outlines_shp, driver='ESRI Shapefile')

    grid_info: gisutil.RasterInfo
        Definition of the grid used for fjord
        Eg: gisutil.RasterInfo(grid_file)

    Returns: np.ndarray
        Mask equals 1 inside the polygons, and 0 outside.
    """

    # http://www2.geog.ucl.ac.uk/~plewis/geogg122_current/_build/html/ChapterX_GDAL/OGR_Python.html

    # Reproject original polygon file to a new (internal) dataset
    # src_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource('x.shp')
    src_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
    srs = osr.SpatialReference(wkt=grid_info.wkt)
    ogrutil.reproject(polygon_ds, srs, src_ds)
    src_lyr = src_ds.GetLayer()   # Put layer number or name in here

    # Create destination raster dataset
#    dst_ds = clone_geometry('netCDF', 'x.nc', grid_info, 1,  gdal.GDT_Byte)
    dst_ds = gdalutil.clone_geometry('MEM', '', grid_info, 1,  gdal.GDT_Byte)
    dst_rb = dst_ds.GetRasterBand(1)
    dst_rb.Fill(0) #initialise raster with zeros
    dst_rb.SetNoDataValue(0)

    maskvalue = 1
    bands = [1]          # Bands to rasterize into
    burn_values = [1]    # Burn this value for each band
    gdalutil.check_error(gdal.RasterizeLayer(dst_ds, bands, src_lyr, burn_values=burn_values))

    dst_ds.FlushCache()

    mask_arr = dst_ds.GetRasterBand(1).ReadAsArray()
    return mask_arr

def rasterize_polygon_compressed(poly, grid_info, debug=False):
    """Burns a polygon into a raster and returns a list of the
    gridcells that intersect with the interior of the polygon.

    polygon_ds:
        Open OGR dataset containing polygons in a single layer
        Can be Shapefile, GeoJSON, etc.
        Eg:
            poly_ds = gdalutil.open(outlines_shp, driver='ESRI Shapefile')
            poly_ds = shapelyutil.to_datasource(shapely_polygon)
    grid_info: gdal_util.grid_info()
        Definition of the receiving raster for the burn
    Returns: [ix, ...]
        The one-dimensional index of each burned gridcell.
        Gridcells are assumed to be indexed as (j, i): the j (y)
        dimension has largest stride.

    """

    # ---------------------------------------------------
    # Burn on entire raster ("Slow Burn")
    if debug:
        poly_ds = shapelyutil.to_datasource(poly)
        try:
            pra0_ras = rasterize_polygons(poly_ds, grid_info)
        finally:
            poly_ds = None    # Free memory

        # Convert to a list of initial gridcell IDs
        jarr0, iarr0 = np.where(pra0_ras)
        jarr0 = jarr0.astype('i')
        iarr0 = iarr0.astype('i')

        pra0_ras1d = pra0_ras.reshape(-1)
        pra0_burn_a = np.where(pra0_ras1d)[0].astype('i')
        pra0_burn_b = jarr0 * grid_info.nx + iarr0
        assert np.all(pra0_burn_a == pra0_burn_b)  # Get our indexing correct

        print('iarr0 = ', iarr0[:10])
        print('jarr0 = ', jarr0[:10])

    # ------------ Work in a smaller coord system ("Fast Burn")
    # Get oriented minimum bounding rectangle (MBR)
    xx,yy = poly.exterior.coords.xy
    minx = np.min(xx)
    maxx = np.max(xx)
    miny = np.min(yy)
    maxy = np.max(yy)
    if debug:
        print('poly bounds: ', minx, maxx, miny, maxy)

    # Extent of the polygon in pixels
    margin = 2
    mini,minj = grid_info.to_ij(minx,miny)
    maxi,maxj = grid_info.to_ij(maxx,maxy)
    if debug:
        print('poly ibounds: ', mini, maxi, minj, maxj)
        print('grid_info.nx/ny: ', grid_info.nx, grid_info.ny)
        print('grid_info.dx/dy: ', grid_info.dx, grid_info.dy)
    origin_i = mini-margin if grid_info.dx > 0 else maxi-margin
    origin_j = minj-margin if grid_info.dy > 0 else maxj-margin
    origin_x,origin_y = grid_info.to_xy(origin_i, origin_j)

    #if debug:
    #    print('pra ', pra)
    #    print('origin_xy ', origin_x, origin_y)

    # Size to make new grid: put 2-pixel margen on all sides
    nx1 = abs(maxi-mini) + margin*2
    ny1 = abs(maxj-minj) + margin*2

    if debug:
        print(f'origin_xy=({origin_x}, {origin_y}); nx/ny=({nx1},{ny1})')
        print(f'origin_ij=({origin_i}, {origin_j})')


    # Define new grid_info for smaller coord system
    gt1 = np.array(grid_info.geotransform)
    gt1[0] = origin_x
    gt1[3] = origin_y
    if debug:
        print('gt-diff x: ', gt1[0] - grid_info.geotransform[0])
        print('gt-diff y: ', gt1[3] - grid_info.geotransform[3])
    grid_info1 = gisutil.RasterInfo(
        grid_info.wkt, nx1, ny1, gt1)

    # ---------- Now working in sub-coord system (grid_info1 / pra1)
    # Burn the PRA polygon into a raster
    # pra1_ras is np.array(nj, ni)

    # NOTE: shapely.to_datasource() must be run EVERY TIME the
    #       datasource is used in rasterize_polygons.  Re-using the
    #       previous value of poly_ds does NOT work.
    poly_ds = shapelyutil.to_datasource(poly)
    try:
        pra1_ras = rasterize_polygons(poly_ds, grid_info1)
    finally:
        poly_ds = None    # Free memory

    #print('pra1_ras\n', pra1_ras)
    #print('yyyy ', grid_info.nx, grid_info.ny)
    if debug:
        print('burn-size: {} vs {}'.format(np.sum(pra1_ras), np.sum(pra0_ras)))
        print('pra1_ras shape:', pra1_ras.shape)
        #print(pra1_ras)
        assert np.sum(pra1_ras) == np.sum(pra0_ras)

    # Get x and y coordinates of burnt pixels (two numpy arrays of indices)
    jarr1, iarr1 = np.where(pra1_ras)
    jarr1 = jarr1.astype('i')
    iarr1 = iarr1.astype('i')

    #if debug:
        #print('iarr0 ', iarr0)
        #print('iarr1 ', iarr1 + origin_i)
        #print('jarr0 ', jarr0)
        #print('jarr1 ', jarr1 + origin_j)


    pra1_burn = (jarr1+origin_j) * grid_info.nx + (iarr1 + origin_i)
    if debug:
        print('pra0_burn_a ', pra0_burn_a)
        print('pra1_burn ', pra1_burn)
        assert np.all(pra0_burn_a == pra1_burn)

    return pra1_burn


