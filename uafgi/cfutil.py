import re
import cf_units
from uafgi import ncutil

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

