import cf_units
import re
import pytest

def test_cf_relative_time_conversion():
    """Demonstrates how to convert "xxx since yyy" units to seconds"""

    cfdateRE = re.compile(r'([^\s]*)(\s+since\s+.*)')
    def unit_to_seconds(sunit):
        match = cfdateRE.match(sunit)
        return 'seconds'+match.group(2)

    sunit = 'days since 2000-01-01'
    udays = cf_units.Unit(sunit)
    usecs = cf_units.Unit(unit_to_seconds(sunit))

    d = 2
    s = udays.convert(d, usecs)
    assert s == d*86400

