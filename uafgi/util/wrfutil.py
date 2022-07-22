import pyproj
import numpy as np
import netCDF4
from uafgi.util import gisutil
import dggs.data

def wrf_info(geo_fname):
    """
    geo_fname:
        Name of the WRF geometry definition file (eg: geosoutheast.nc)
    """
    with netCDF4.Dataset(geo_fname) as nc:
        # Get lon/lat of center of each gridcell
        lon_m = nc.variables['XLONG_M'][0,:,:]
        lat_m = nc.variables['XLAT_M'][0,:,:]
        nji = lon_m.shape

        # NOTE: WRF NetCDF file is in north-down format; i.e. the most
        #       northernly points are at the END of the Y axis.  This
        #       is the opposite of typical GeoTIFF, and we would
        #       expect dy to be positive.

        # Obtain CRS for this WRF run
        MAP_PROJ = nc.MAP_PROJ
        if MAP_PROJ != 1:
            raise ValueError('WRF MAP_PROJ={} must be 1 (Lambert Concical LCC projection in PROJ.4).  Other values of MAP_PROJ are not supported'.format(MAP_PROJ))
        wrf_crs = pyproj.CRS.from_string(f'+proj=lcc +lat_1={nc.TRUELAT1} +lat_2={nc.TRUELAT2} +lat_0={nc.CEN_LAT} +lon_0={nc.CEN_LON} +x_0=0 +y_0=0 +a=6370000 +b=6370000 +units=m +no_defs')


    # Convert Gridcell centers from lon/lat to WRF's CRS
    lonlat_crs = pyproj.CRS.from_string(dggs.data.grs1980_wkt)
    ll2wrf = pyproj.Transformer.from_crs(lonlat_crs, wrf_crs, always_xy=True)
    xx_m_wrf, yy_m_wrf = ll2wrf.transform(lon_m,lat_m)

    # Set up the geotransform for the WRF raster
    # See: https://gdal.org/tutorials/geotransforms_tut.html
    dx = np.mean(xx_m_wrf[:,1:] - xx_m_wrf[:,0:-1])
    dy = np.mean(yy_m_wrf[1:,:] - yy_m_wrf[0:-1,:])
    gt_wrf = [
        np.mean(xx_m_wrf[:,0]) - .5*dx,    # edge of x-coord of origin (most westerly pixel)
        dx,                       # W-E pixel width
        0,                        # Row rotation
        np.mean(yy_m_wrf[0,:]) + .5*dy,    # edge of y-coord of origin (most southerly pixel in this case)
        0,                        # Column rotation

        dy]                       # N-S pixel resolution (negative val for north-up images)

    return gisutil.RasterInfo(wrf_crs.to_wkt(), nji[1], nji[0], gt_wrf)


#class WRFTransformer:
#    def __init__(self, geo_fname, scene_wkt):
#
#        self.info = wrf_info(geo_fname)
#
#        # Obtain transfomer from scene coordinates to what WRF uses.
#        scene_crs = pyproj.CRS.from_string(scene_wkt)
#        wrf_crs = pyproj.CRS.from_string(self.info.wkt)
#        # There will be "error" in this because the spheroids do not match.
#        # WRF uses perfect sphere; whereas scene typically uses WGS84 or similar
#        self.scene2wrf = pyproj.Transformer.from_crs(scene_crs, wrf_crs, always_xy=True)
#
#    def to_ij(self, xx_scene, yy_scene):
#        """xx_scene, yy_scene:
#            Coordinates to convert, IN SCENE SPACE"""
#        xx_wrf,yy_wrf = self.scene2wrf.transform(xx_scene, yy_scene)
#        return self.info.to_ij(xx_wrf, yy_wrf)
#
#
