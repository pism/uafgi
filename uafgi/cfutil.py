import netCDF4
import bisect
import re
import cf_units
from uafgi import ncutil
import datetime
import numpy as np

def get_crs(cf_file):
    """Returns a PROJ CRS obtained from a CF-compliant NetCDF file.

    NOTE: Currently only works if the variable "polar_stereographic" exists.

    cf_file:
        Filename or open NetCDF file handle.
    """
    with ncutil.reopen_nc(cf_file) as nc:
        wks_s = nc.variables['polar_stereographic'].spatial_ref

    return pyproj.CRS.from_string(wks_s)


cfdateRE = re.compile(r'([^\s]*)(\s+since\s+.*)')
def replace_reftime_unit(unit, relname='seconds'):
    """Given a reftime unit (eg: 'days since 2000-12-01'), creates a new unit
    with the same reference time but different units (eg: 'minutes since 2000-12-01')

    relname:
        Name of new relative unit to use (eg: 'seconds', 'days', etc)
    """

    if not unit.is_time_reference():
        raise ValueError('Requires reftime unit: eg "days since 2000-12-1"')

    match = cfdateRE.match(unit.origin)    # string representation of unit
    return cf_units.Unit(relname+match.group(2), unit.calendar)


def read_time(nc, vname, units=None, calendar=None, unitvar=None):
    """Reads a CF-compliant time variable and converts to Python datetime objects.
    nc: netCDF4.Dataset
        An open NetCDF file
    vname: str
        Name of time variable to read
    unitvar: str
        Obtain unit variables from this variable
    Returns: [datetime, ...]
        The times, converted to Python format.
    """
    # Read units from unitvar, if supplied
    if unitvar is not None:
        uvar = nc.variables[unitvar]
        units = uvar.units
        calendar = uvar.calendar

    nctime = nc.variables[vname]
    if units is None:
        units = nctime.units
    if calendar is None:
        calendar = nctime.calendar

    ret = list()
    n2d = cf_units.Unit(units, calendar=calendar).num2date(nctime[:])

    # Convert from CFUnit datetime to standard Python datetime
    ret = np.vectorize(
        lambda dt: datetime.datetime(
            dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        )(n2d)
    return ret

def convert(ncarray, from_unit, to_unit):
    """Converts physical units.  Convenience function.
    ncarray:
        The data to convert.
    from_unit: str or cf_units.Unit
        The units ncarray is in
    to_unit: str or cf_units.Unit
        The units we want
    Returns:
        Converted data
    """
    if isinstance(from_unit,str):
        from_unit = cf_units.Unit(from_unit)
    if isinstance(to_unit,str):
        to_unit = cf_units.Unit(to_unit)

    return from_unit.convert(ncarray, to_unit)

def index_slice(vals, v0,v1):
    """Finds a range of indices [i0,i1) that encompases the range of values [v0,v1]
    Returns: i0,i1"""
    i0 = bisect.bisect_left(vals,v0)
    i1 = bisect.bisect_right(vals,v1)
    return slice(i0,i1)


class LonLatSubGrid:
    """Defines items of LonLat grid to help write CF-compliant files, based on the grid cell centers."""

    def __init__(self, lon_range, lat_range, longitude=None, latitude=None, sample_file=None):
        """
        lon_range: [min, max)
            Longitude range to include in the output
        lat_range: [min, max)
            Latitude range to include in the output
        longitude:
            List/array of grid cell centers for the global grid
        latitude:
            List/array of grid cell centers for the global grid
        sample_file:
            (OPTIONAL) Get longitude and latitude from this file, then don't have to specify.
        """

        self.lon_range = lon_range
        self.lat_range = lat_range

        self.longitude = longitude
        self.latitude = latitude
        if sample_file is not None:
            with netCDF4.Dataset(sample_file) as nc:
                nc.set_always_mask(False)
                self.longitude = nc.variables['longitude'][:]
                self.latitude = nc.variables['latitude'][:]

        # Determine area slices and overall dimensions
        self.lonslice = index_slice(self.longitude, *lon_range)
        self.nlon = len(range(*self.lonslice.indices(len(self.longitude))))
        self.lon = self.longitude[self.lonslice]


        self.latslice = index_slice(-self.latitude, -lat_range[1], -lat_range[0])  # Negation because latitude is highest to lowest
        self.nlat = len(range(*self.latslice.indices(len(self.latitude))))
        self.lat = self.latitude[self.latslice]

        # https://gdal.org/tutorials/geotransforms_tut.html
        self.dlon = self.lon[1] - self.lon[0]
        self.dlat = self.lat[1] - self.lat[0]
        self.geotransform = [
            self.lon[0] - .5*self.dlon, self.dlon, 0,
            self.lat[0] - .5*self.dlat, 0, self.dlat]

def write_wgs84(nc, geotransform, vname='crs'):

    """Writes a CRS variable into the NetCDF file.  Other variables can
    reference it with the `grid_mapping` attribute.

    geotransform:
        Standard six items in list.
        See LonLatSubGrid above

    Example:
    ```
        char crs ;
                crs:grid_mapping_name = "latitude_longitude" ;
                crs:long_name = "CRS definition" ;
                crs:longitude_of_prime_meridian = 0. ;
                crs:semi_major_axis = 6378137. ;
                crs:inverse_flattening = 298.257223563 ;
                crs:spatial_ref = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]" ;
                crs:GeoTransform = "-0.125 0.25 0 75.125 0 -0.25" ;
        double tin(time, latitude, longitude) ;
                tin:units = "degC" ;
                tin:long_name = "2 metre temperature" ;
                tin:grid_mapping = "crs" ;
    ```"""
    nccrs = nc.createVariable(vname, 'c')
    # Standard WGS84 stuff
    nccrs.grid_mapping_name = "latitude_longitude" ;
    nccrs.long_name = "CRS definition" ;
    nccrs.longitude_of_prime_meridian = 0. ;
    nccrs.semi_major_axis = 6378137. ;
    nccrs.inverse_flattening = 298.257223563 ;
    nccrs.spatial_ref = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]" ;
    # Specific to this file
    nccrs.GeoTransform = ' '.join(str(x) for x in geotransform)

    return nccrs
