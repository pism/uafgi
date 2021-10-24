import pyproj
import cartopy.crs
import numpy as np
import shapely.geometry

def _ellipse_boundary(semimajor=2, semiminor=1, easting=0, northing=0, n=201):
    """
    Define a projection boundary using an ellipse.

    This type of boundary is used by several projections.

    """

    t = np.linspace(0, -2 * np.pi, n)  # Clockwise boundary.
    coords = np.vstack([semimajor * np.cos(t), semiminor * np.sin(t)])
    coords += ([easting], [northing])
    return coords


class Stereographic(cartopy.crs.Projection):
    def __init__(self, proj4_dict, globe=None):
        super().__init__(list(proj4_dict.items()), globe=globe)

        # TODO: Get these out of proj4_dict
        false_easting = proj4_dict['x_0']
        false_northing = proj4_dict['y_0']

        # TODO: Let the globe return the semimajor axis always.
        a = np.float(self.globe.semimajor_axis or cartopy.crs.WGS84_SEMIMAJOR_AXIS)
        b = np.float(self.globe.semiminor_axis or cartopy.crs.WGS84_SEMIMINOR_AXIS)

        # Note: The magic number has been picked to maintain consistent
        # behaviour with a wgs84 globe. There is no guarantee that the scaling
        # should even be linear.
        x_axis_offset = 5e7 / cartopy.crs.WGS84_SEMIMAJOR_AXIS
        y_axis_offset = 5e7 / cartopy.crs.WGS84_SEMIMINOR_AXIS
        self._x_limits = (-a * x_axis_offset + false_easting,
                          a * x_axis_offset + false_easting)
        self._y_limits = (-b * y_axis_offset + false_northing,
                          b * y_axis_offset + false_northing)
        coords = _ellipse_boundary(self._x_limits[1], self._y_limits[1],
                                   false_easting, false_northing, 91)
        self._boundary = shapely.geometry.LinearRing(coords.T)
        self._threshold = np.diff(self._x_limits)[0] * 1e-3

    @property
    def boundary(self):
        return self._boundary

    @property
    def threshold(self):
        return self._threshold

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits

#
#class _RectangularProjection(cartopy.crs.Projection):
#    """
#    The abstract superclass of projections with a rectangular domain which
#    is symmetric about the origin.
#
#    """
#    def __init__(self, proj4_dict, half_width, half_height, globe=None):
#        self._half_width = half_width
#        self._half_height = half_height
#        super().__init__(list(proj4_dict.items()), globe=globe)
#
#    @property
#    def boundary(self):
#        # XXX Should this be a LinearRing?
#        w, h = self._half_width, self._half_height
#        return sgeom.LineString([(-w, -h), (-w, h), (w, h), (w, -h), (-w, -h)])
#
#    @property
#    def x_limits(self):
#        return (-self._half_width, self._half_width)
#
#    @property
#    def y_limits(self):
#        return (-self._half_height, self._half_height)
#
#

# -------------------------------------------------------------------

_cartopy_proj_classes = {
    'stere': Stereographic,
}

# PROJ params to be used to  construct the Globe;
# and corresponding Cartopy Globe params
_GLOBE_PARAMS = {'datum': 'datum',
                 'ellps': 'ellipse',
                 'a': 'semimajor_axis',
                 'b': 'semiminor_axis',
                 'f': 'flattening',
                 'rf': 'inverse_flattening',
                 'towgs84': 'towgs84',
                 'nadgrids': 'nadgrids'}


def crs(projparams):
    """Constructs a Cartopy projection based on PyProj parameters.
    projparams: {key: value, ...}
        Stuff used to initialize pyproj.crs.CRS()
        See: https://pyproj4.github.io/pyproj/dev/api/crs/crs.html
    """

    ppcrs = pyproj.crs.CRS(projparams)
    ppdict = ppcrs.to_dict()

    # Split and translate PyProj parameters
    globe_params = dict()
    other_params = dict()
    for ppkey,val in ppdict.items():
        if ppkey in _GLOBE_PARAMS:
            globe_params[_GLOBE_PARAMS[ppkey]] = val
        else:
            other_params[ppkey] = val

    # Construct Cartopy CRS from the PyProj key/value pairs
    globe = cartopy.crs.Globe(**globe_params)
    try:
        cartopy_klass = _cartopy_proj_classes[ppdict['proj']]
    except:
        raise NotImplementedError("Cartopy Projection subclass for PROJ type '{}' not yet implemented.".format(ppdict['proj'])) from None

    cartopy_crs = cartopy_klass(other_params, globe=globe)
    return cartopy_crs
