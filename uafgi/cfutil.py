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
