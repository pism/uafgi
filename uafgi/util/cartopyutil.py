import collections
import pyproj
import numpy as np
import shapely.geometry
import cartopy.crs
import cartopy.geodesic
import matplotlib.pyplot as plt
import matplotlib.font_manager
from uafgi.util import gdalutil

def _ellipse_boundary(semimajor=2, semiminor=1, easting=0, northing=0, n=201):
    """
    Define a projection boundary using an ellipse.

    This type of boundary is used by several projections.

    """

    t = np.linspace(0, -2 * np.pi, n)  # Clockwise boundary.
    coords = np.vstack([semimajor * np.cos(t), semiminor * np.sin(t)])
    coords += ([easting], [northing])
    return coords

# --------------------------------------------------------
# These classes are copied from cartopy/crs.py, then modified to take
# inputs from proj4_dict.


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

class AlbersEqualArea(cartopy.crs.Projection):
    """
    An Albers Equal Area projection

    This projection is conic and equal-area, and is commonly used for maps of
    the conterminous United States.

    """

    def __init__(self, proj4_dict, globe=None):

        """
        Parameters
        ----------
        central_longitude: optional
            The central longitude. Defaults to 0.
        central_latitude: optional
            The central latitude. Defaults to 0.
        false_easting: optional
            X offset from planar origin in metres. Defaults to 0.
        false_northing: optional
            Y offset from planar origin in metres. Defaults to 0.
        standard_parallels: optional
            The one or two latitudes of correct scale. Defaults to (20, 50).
        globe: optional
            A :class:`cartopy.crs.Globe`. If omitted, a default globe is
            created.

        """
        super().__init__(list(proj4_dict.items()), globe=globe)

        central_longitude = proj4_dict['lon_0']

        # bounds
        minlon, maxlon = self._determine_longitude_bounds(central_longitude)
        n = 103
        lons = np.empty(2 * n + 1)
        lats = np.empty(2 * n + 1)
        tmp = np.linspace(minlon, maxlon, n)
        lons[:n] = tmp
        lats[:n] = 90
        lons[n:-1] = tmp[::-1]
        lats[n:-1] = -90
        lons[-1] = lons[0]
        lats[-1] = lats[0]

        points = self.transform_points(self.as_geodetic(), lons, lats)

        self._boundary = shapely.geometry.LinearRing(points)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        self._x_limits = mins[0], maxs[0]
        self._y_limits = mins[1], maxs[1]

        self.threshold = 1e5

    @property
    def boundary(self):
        return self._boundary

    @property
    def x_limits(self):
        return self._x_limits

    @property
    def y_limits(self):
        return self._y_limits


class LambertConformal(cartopy.crs.Projection):
    """
    A Lambert Conformal conic projection.

    """

    def __init__(self, proj4_dict, globe=None, cutoff=-30):
#self, central_longitude=-96.0, central_latitude=39.0,
#                 false_easting=0.0, false_northing=0.0,
#                 standard_parallels=(33, 45),
#                 globe=None, cutoff=-30):
        """
        Parameters
        ----------
        central_longitude: optional
            The central longitude. Defaults to -96.
        central_latitude: optional
            The central latitude. Defaults to 39.
        false_easting: optional
            X offset from planar origin in metres. Defaults to 0.
        false_northing: optional
            Y offset from planar origin in metres. Defaults to 0.
        standard_parallels: optional
            Standard parallel latitude(s). Defaults to (33, 45).
        globe: optional
            A :class:`cartopy.crs.Globe`. If omitted, a default globe is
            created.
        cutoff: optional
            Latitude of map cutoff.
            The map extends to infinity opposite the central pole
            so we must cut off the map drawing before then.
            A value of 0 will draw half the globe. Defaults to -30.

        """
        super().__init__(list(proj4_dict.items()), globe=globe)
#        proj4_params = [('proj', 'lcc'),
#                        ('lon_0', central_longitude),
#                        ('lat_0', central_latitude),
#                        ('x_0', false_easting),
#                        ('y_0', false_northing)]
#
#        n_parallels = len(standard_parallels)
#
#        if not 1 <= n_parallels <= 2:
#            raise ValueError('1 or 2 standard parallels must be specified. '
#                             f'Got {n_parallels} ({standard_parallels})')
#
#        proj4_params.append(('lat_1', standard_parallels[0]))
#        if n_parallels == 2:
#            proj4_params.append(('lat_2', standard_parallels[1]))
#
#        super().__init__(proj4_params, globe=globe)

        standard_parallels = (proj4_dict['lat_1'], proj4_dict['lat_2'])
        central_longitude = proj4_dict['lon_0']

        n_parallels = len(standard_parallels)

        # Compute whether this projection is at the "north pole" or the
        # "south pole" (after the central lon/lat have been taken into
        # account).
        if n_parallels == 1:
            plat = 90 if standard_parallels[0] > 0 else -90
        else:
            # Which pole are the parallels closest to? That is the direction
            # that the cone converges.
            if abs(standard_parallels[0]) > abs(standard_parallels[1]):
                poliest_sec = standard_parallels[0]
            else:
                poliest_sec = standard_parallels[1]
            plat = 90 if poliest_sec > 0 else -90

        self.cutoff = cutoff
        n = 91
        lons = np.empty(n + 2)
        lats = np.full(n + 2, float(cutoff))
        lons[0] = lons[-1] = 0
        lats[0] = lats[-1] = plat
        if plat == 90:
            # Ensure clockwise
            lons[1:-1] = np.linspace(central_longitude + 180 - 0.001,
                                     central_longitude - 180 + 0.001, n)
        else:
            lons[1:-1] = np.linspace(central_longitude - 180 + 0.001,
                                     central_longitude + 180 - 0.001, n)

        points = self.transform_points(self.as_geodetic(), lons, lats)

        self._boundary = shapely.geometry.LinearRing(points)
        mins = np.min(points, axis=0)
        maxs = np.max(points, axis=0)
        self._x_limits = mins[0], maxs[0]
        self._y_limits = mins[1], maxs[1]

        self.threshold = 1e5

    def __eq__(self, other):
        res = super().__eq__(other)
        if hasattr(other, "cutoff"):
            res = res and self.cutoff == other.cutoff
        return res

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.proj4_init, self.cutoff))

    @property
    def boundary(self):
        return self._boundary

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
    'aea': AlbersEqualArea,
    'lcc': LambertConformal,
}

# PROJ params to be used to  construct the Globe;
# and corresponding Cartopy Globe params
_GLOBE_PARAMS = {'datum': 'datum',
                 'ellps': 'ellipse',
                 'a': 'semimajor_axis',
                 'b': 'semiminor_axis',
                 'f': 'flattening',
                 'rf': 'inverse_flattening',
                 'towgs84': 'towgs84',    # Datum transformation to WGS84
                 'nadgrids': 'nadgrids'}


def crs(projparams):
    """Constructs a Cartopy projection based on PyProj parameters.
    projparams: {key: value, ...}
        Stuff used to initialize pyproj.crs.CRS()
        See: https://pyproj4.github.io/pyproj/dev/api/crs/crs.html
    """

#    print('crs projparams: ', projparams, type(projparams))
    ppcrs = pyproj.crs.CRS(projparams)
    ppdict = ppcrs.to_dict()
#    print('crs ppdict: ', ppdict)
##    ppdict['towgs84'] = '500000,0,0,0,0,0,1000000'    # DEBUGGING: This changes things!!!

    # Split and translate PyProj parameters
    globe_params = dict()
    other_params = dict()
    for ppkey,val in ppdict.items():
        if ppkey in _GLOBE_PARAMS:
            globe_params[_GLOBE_PARAMS[ppkey]] = val
        else:
            other_params[ppkey] = val

    # Construct Cartopy CRS from the PyProj key/value pairs
#    print('globe_params ', globe_params)
#    print('other_params ', other_params)

    globe = cartopy.crs.Globe(**globe_params)
    try:
        cartopy_klass = _cartopy_proj_classes[ppdict['proj']]
    except:
        raise NotImplementedError("Cartopy Projection subclass for PROJ type '{}' not yet implemented.".format(ppdict['proj'])) from None

#    other_params['towgs84'] = [500000,0,0,0,0,0,1000000]    # This does not change things
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

_at_y_trans = {
    'top': (0.89, 0.915),
    'bottom': (0.08, 0.105)
}

#def add_osgb_scalebar(ax, at_x=(0.1, 0.4), at_y=(0.06, 0.075), max_stripes=5, text_color='black'):
def add_osgb_scalebar(ax, at_x=(0.1, 0.35), at_y='bottom', max_stripes=5, text_color='black', alpha=0.3):
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

    at_y = _at_y_trans.get(at_y, at_y)

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
        # fill region (but don't fill any white)
        if i_color == 0:
            plt.fill((xi0, xi1, xi1, xi0, xi0), (yl0, yl0, yl1, yl1, yl0),
                 fill_colors[i_color], alpha=alpha)
        # draw boundary
        plt.plot((xi0, xi1, xi1, xi0, xi0), (yl0, yl0, yl1, yl1, yl0),
                 'black', alpha=.8)
        i_color = 1 - i_color

    # add short tick lines
    for x in x_vals:
        plt.plot((x, x), (yl0, yl0-y_margin), 'black')

    # add a scale legend 'Km'
    font_props = matplotlib.font_manager.FontProperties(size='medium', weight='bold')
    plt.text(
        0.5 * (xl0 + xl1),
        yl1 + y_margin,
        'km',
        color=text_color,
        verticalalignment='bottom',
        horizontalalignment='center',
        fontproperties=font_props)

    # add numeric labels
#    for x in x_vals:
    for x in (x_vals[0], x_vals[-1]):
        plt.text(x,
                 yl0 - 2 * y_margin,
                 '{:g}'.format((x - xl0) * 0.001),    # *.001 = km
                 color=text_color,
                 verticalalignment='top',
                 horizontalalignment='center',
                 fontproperties=font_props)


# --------------------------------------------------------------------
def _xy_extents(geotransform, nx, ny):
    """Compute Cartopy map extents from input geotransform"""
    x0 = geotransform[0]
    x1 = x0 + geotransform[1] * nx
    y0 = geotransform[3]
    y1 = y0 + geotransform[5] * ny
    extents = [x0,x1,y0,y1]
    return extents

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
    extents = _xy_extents(geotransform, nx, ny)

    return MapInfo(map_crs, extents)
#    ax.set_extent(extents=extents, crs=map_crs)
#    return map_crs


def raster_mapinfo(raster_file):
    """
    raster_file:
        Name of a GeoTIFF or other GDAL-readable raster file
    """
    raster = gdalutil.read_raster(raster_file, data=False)
    map_crs = crs(raster.grid.wkt)

    extents = _xy_extents(raster.grid.geotransform, raster.grid.nx, raster.grid.ny)
    return MapInfo(map_crs, extents)
# --------------------------------------------------
def plot_hillshade(ax, dem_data, extent=None, transform=None, cmap='Greys', **kwargs):
    """Plots hillshade (shaded relief) map
    ax:
        Matplotlib / Cartopy object to plot it on
    dem_data:
        A digital elevation model (DEM) to plot, as with pcolormesh.
    extent:
        The bounds of dem_data (XXYY order)
    transform:
        The CRS used for dem_data
    """

    x,y = np.gradient(dem_data)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))

    # -x here because of pixel orders in the SRTM tile
    aspect = np.arctan2(-x, y)

    altitude = np.pi / 4.
    azimuth = np.pi / 2.

    shaded = np.sin(altitude) * np.sin(slope)\
        + np.cos(altitude) * np.cos(slope)\
        * np.cos((azimuth - np.pi/2.) - aspect)
    return plt.imshow(shaded, extent=extent, transform=transform,
        cmap=cmap, **kwargs)
# -----------------------------------------------------------------
def poly_clip_path(ax, mpoly):
    """
    originfig:
        The plot element to be clipped.
        Eg:
            cf = ax.contourf(t2.lon, t2.lat, t2, extend = 'both',  transform = proj)
            clip_drawing(cf, ax, mpl)
    ax:
        Standard Cartopy axes, eg:
            fig = plt.figure(figsize=(6,6)) 
            ax = fig.subplots(1, 1, subplot_kw={'projection': proj})  

    mpoly: shapely.*.MultiPolygon
        Clip to this shape
    """
    vertices = []
    codes = []

    if isinstance(mpoly, shapely.geometry.polygon.Polygon):
        polys = (mpoly,)
    elif isinstance(mpoly, shapely.geometry.multipolygon,MultiPolygon):
        polys = list(mpoly.geoms)
    else:
        raise TypeError('clip_to_poly needs a Polygon or MultiPolygon')

    for poly in polys:
        xys = poly.exterior.coords
        vertices += list(xys)    # [(x,y), ...]
        codes += [matplotlib.patches.Path.MOVETO]
        codes += [matplotlib.patches.Path.LINETO]*(len(xys)-2)
        codes += [matplotlib.patches.Path.CLOSEPOLY]
    clip = matplotlib.patches.Path(vertices, codes)
    clip = matplotlib.patches.PathPatch(clip, transform=ax.transData)

    return clip

#    originfig.set_clip_path(clip)
#
##    for contour in originfig.collections:
##        contour.set_clip_path(clip)
#    return clip
