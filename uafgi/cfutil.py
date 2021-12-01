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


class LonLatGrid:
    """Defines items of global LonLat grid to help write CF-compliant files,
    based on the grid cell centers."""

    def __init__(self, longitude, latitude)

        self.lon = longitude
        self.lat = latitude

        # https://gdal.org/tutorials/geotransforms_tut.html
        self.dlon = self.lon[1] - self.lon[0]
        self.dlat = self.lat[1] - self.lat[0]
        self.geotransform = [
            self.lon[0] - .5*self.dlon, self.dlon, 0,
            self.lat[0] - .5*self.dlat, 0, self.dlat]

    def subgrid(self, lon_range=(0,360), lat_range=(-90,90)):
        """Returns a LonLatGrid that's a sub-grid of the current one."""

        # Determine area slices and overall dimensions
        lonslice = index_slice(self.lon, *lon_range)
        latslice = index_slice(-self.lat, -lat_range[1], -lat_range[0])  # Negation because latitude is highest to lowest

        return LonLatGrid(self.lon[lonslice], self.lat[latslice])


    @property
    def nlat(self):
        return len(self.lat))
    @property
    def nlon(self):
        return len(self.lon))

    def ncdef(self, nc, crs_vname='crs'):
        """To set up a CF-compliant CRS in your NetCDF file, first call ncdef(),
        then ncwrite().
        """
        nc.createDimension('longitude', self.nlon)
        nc.createDimension('latitude', self.nlat)
        nclon = nc.createVariable('longitude', 'd', ('longitude',))
        nclat = nc.createVariable('latitude', 'd', ('latitude',))
        nccrs = nc.createVariable(crs_vname, 'c')
        ncutil.setncattrs(nccrs, self.wgs84_cf_attrs)

    def ncwrite(self, nc, crs_vname='crs'):
        nc.variables['longitude'][:] = self.lon
        nc.variables['latitude'][:] = self.lat

    @property
    def wgs84_cf_attrs(self):
        """Returns a dict of the attributes needed for a NetCDF CF-Compliant
        CRS definition variable.

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


        # Standard WGS84 stuff
        return {
            'grid_mapping_name': "latitude_longitude" ;
            'long_name': "CRS definition" ;
            'longitude_of_prime_meridian': 0. ;
            'semi_major_axis': 6378137. ;
            'inverse_flattening': 298.257223563 ;
            'spatial_ref': "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]" ;
            # Specific to this file
            'GeoTransform': ' '.join(str(x) for x in self.geotransform)
        }


def lonlat_from_samplefile(sample_file):
    """Creates a LonLatGrid based on the longitude and latitude variables
    in an existing file (does not need any other CF-cocmpliant stuff)"""

    # Read longitude and latitude from sample file
    with netCDF4.Dataset(sample_file) as nc:
        nc.set_always_mask(False)
        longitude = nc.variables['longitude'][:]
        latitude = nc.variables['latitude'][:]

    return LonLatGrid(longitude, latitude)

def lonlat_by_size(nlon, nlat):

    """Creates a global LonLatGrid based on the number of longitude and
    latitude points (and some assumptions)"""
    lon = np.linspace(0,360,num=nlon, endpoint=False)
    lat = np.linspace(-90,90,num=nlat, endpoint=True)

    return LonLatGrid(lon, lat)
