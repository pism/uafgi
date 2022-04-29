import cartopy






# -------------------------------------------------
class Stereographic(cartopy.crs.Projection):

    """A re-do of cartopy.crs.Stereographic; but it takes proj4_terms
    rather than its own bespoke kwargs."""

    def __init__(self, proj4_terms, globe=None):
        super(Stereographic, self).__init__(proj4_terms, globe=globe)

        # TODO: Get these out of proj4_params
        false_easting = 0
        false_northing = 0

        # TODO: Let the globe return the semimajor axis always.
        a = np.float(self.globe.semimajor_axis or WGS84_SEMIMAJOR_AXIS)
        b = np.float(self.globe.semiminor_axis or WGS84_SEMIMINOR_AXIS)

        # Note: The magic number has been picked to maintain consistent
        # behaviour with a wgs84 globe. There is no guarantee that the scaling
        # should even be linear.
        x_axis_offset = 5e7 / WGS84_SEMIMAJOR_AXIS
        y_axis_offset = 5e7 / WGS84_SEMIMINOR_AXIS
        self._x_limits = (-a * x_axis_offset + false_easting,
                          a * x_axis_offset + false_easting)
        self._y_limits = (-b * y_axis_offset + false_northing,
                          b * y_axis_offset + false_northing)
        coords = _ellipse_boundary(self._x_limits[1], self._y_limits[1],
                                   false_easting, false_northing, 91)
        self._boundary = sgeom.LinearRing(coords.T)
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




# -------------------------------------------------

# PROJ.4 parames used for the Globe (and not projection per se)
_GLOBE_PARAMS = {'datum': 'datum',
                 'ellps': 'ellipse',
                 'a': 'semimajor_axis',
                 'b': 'semiminor_axis',
                 'f': 'flattening',
                 'rf': 'inverse_flattening',
                 'towgs84': 'towgs84',
                 'nadgrids': 'nadgrids'}


# Assocation between our projection class and PROJ.4 proj=...
_proj4_classes = {
    'stere' : Stereographic,
}


def proj4_crs(proj4_str):
    """Instantiates a Cartopy CRS from a PROJ.4 string.
    Only currently works for certain classes of PROJ.4 strings;
    see _proj4_classes.


    Eg EPSG:3413  WGS 84 / NSIDC Sea Ice Polar Stereographic North
        https://epsg.io/3413
        +proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs 
    """

    terms = [term.strip('+').split('=') for term in proj4_str.split(' ')]
    globe_terms = filter(lambda term: term[0] in _GLOBE_PARAMS, terms)
    #print('globe_terms: {}'.format(list(globe_terms)))
    globe_kwargs = {_GLOBE_PARAMS[name]: value for name, value in globe_terms}
    globe = ccrs.Globe(**globe_kwargs)

    other_terms = []    # Use in Projection.__init__(other_terms, globe)
    for term in terms:
        if term[0] not in _GLOBE_PARAMS:
            if len(term) == 1:
                other_terms.append([term[0], None])
            else:
                other_terms.append(term)
                if (term[0] == 'proj') klass = _proj4_classes[term[1]]

    return klass(other_terms, globe)
# -------------------------------------------------
