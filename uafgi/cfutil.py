import re
import cf_units

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

