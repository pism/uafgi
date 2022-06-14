import collections
import pyproj
import numpy as np
import shapely.geometry
import cartopy.crs
import cartopy.geodesic

import matplotlib.pyplot as plt
import matplotlib.font_manager

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

# ================================================================
# https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot

def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = cartopy.crs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cartopy.geodesic.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
            transform=ax.transAxes, **text_kwargs)


# ========================================================================
# https://nbviewer.org/github/pp-mo/iris_example_code/blob/cartopy_scalebar/map_scalebar.ipynb



def add_osgb_scalebar(ax, at_x=(0.1, 0.4), at_y=(0.05, 0.075), max_stripes=5):
    """
    Add a scalebar to a GeoAxes of type cartopy.crs.OSGB (only).
    NOTE: Coordinates in ax must be in METERS

    Args:
    * at_x : (float, float)
        target axes X coordinates (0..1) of box (= left, right)
    * at_y : (float, float)
        axes Y coordinates (0..1) of box (= lower, upper)
    * max_stripes
        typical/maximum number of black+white regions
    """
    # ensure axis is an OSGB map (meaning coords are just metres)
#    assert isinstance(ax.projection, cartopy.crs.OSGB)
    # fetch axes coordinate mins+maxes
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # set target rectangle in-visible-area (aka 'Axes') coordinates
    ax0, ax1 = at_x
    ay0, ay1 = at_y
    # choose exact X points as sensible grid ticks with Axis 'ticker' helper
    x_targets = [x0 + ax * (x1 - x0) for ax in (ax0, ax1)]
    ll = matplotlib.ticker.MaxNLocator(nbins=max_stripes, steps=[1,2,4,5,10])
    x_vals = ll.tick_values(*x_targets)
    # grab min+max for limits
    xl0, xl1 = x_vals[0], x_vals[-1]
    # calculate Axes Y coordinates of box top+bottom
    yl0, yl1 = [y0 + ay * (y1 - y0) for ay in [ay0, ay1]]
    # calculate Axes Y distance of ticks + label margins
    y_margin = (yl1-yl0)*0.25

    # fill black/white 'stripes' and draw their boundaries
    fill_colors = ['black', 'white']
    i_color = 0
    for xi0, xi1 in zip(x_vals[:-1],x_vals[1:]):
        # fill region
        plt.fill((xi0, xi1, xi1, xi0, xi0), (yl0, yl0, yl1, yl1, yl0),
                 fill_colors[i_color])
        # draw boundary
        plt.plot((xi0, xi1, xi1, xi0, xi0), (yl0, yl0, yl1, yl1, yl0),
                 'black')
        i_color = 1 - i_color

    # add short tick lines
    for x in x_vals:
        plt.plot((x, x), (yl0, yl0-y_margin), 'black')

    # add a scale legend 'Km'
    font_props = matplotlib.font_manager.FontProperties(size='medium', weight='bold')
    plt.text(
        0.5 * (xl0 + xl1),
        yl1 + y_margin,
        'Km',
        verticalalignment='bottom',
        horizontalalignment='center',
        fontproperties=font_props)

    # add numeric labels
    for x in x_vals:
        plt.text(x,
                 yl0 - 2 * y_margin,
                 '{:g}'.format((x - xl0) * 0.001),
                 verticalalignment='top',
                 horizontalalignment='center',
                 fontproperties=font_props)

MapInfo = collections.namedtuple('MapInfo', ('crs', 'extents'))
def nc_mapinfo(nc, ncvarname):
    """Setup a map from CF-compliant stuff"""

    nx = len(nc.dimensions['x'])
    ny = len(nc.dimensions['y'])

    ncvar = nc[ncvarname]
    map_crs = crs(ncvar.spatial_ref)

#    map_crs = cartopy.crs.Stereographic(
#        central_latitude=90,
#        central_longitude=-45,
#        false_easting=0, false_northing=0,
#        true_scale_latitude=70, globe=None)

    print('map_crs ', map_crs)

    # Read extents from the NetCDF geotransform
    geotransform = [float(x) for x in ncvar.GeoTransform.split(' ') if x != '']
    x0 = geotransform[0]
    x1 = x0 + geotransform[1] * nx
    y0 = geotransform[3]
    y1 = y0 + geotransform[5] * ny
    extents = [x0,x1,y0,y1]

    return MapInfo(map_crs, extents)
#    ax.set_extent(extents=extents, crs=map_crs)
#    return map_crs
