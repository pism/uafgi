import os
import subprocess
import netCDF4
from uafgi import ncutil

def fix_output(output_file3, exception, time_units_s, output_file4, delete_vars=list()):
    """Compress PISM output file while correcting time units; and removes original.
    exception:
        Exception (if any) that occurred while running PISM
    units_s: cfunits.Unit
        Units in "seconds since..." format

    NOTE: Fixes to the time unit could be done when the PISM file is opened.
          https://github.com/pism/pism/blob/master/site-packages/PISM/util.py#L30

          To avoid "fixing" the file you could just copy-and-paste the
          guts of prepare_output() and supply your own calendar and
          units string when the file's metadata is written the first
          time.

    """

    # Convert to NetCDF4
    print('Compressing PISM output to NetCDF4...')
    with netCDF4.Dataset(output_file3, 'r') as ncin:
        schema = ncutil.Schema(ncin)

        # Remove excess variables
        print('delete_vars ', delete_vars)
        for vname in delete_vars:
            del schema.vars[vname]

        # Fix time units
        ncv = schema.vars['time']
        ncv.attrs['units'] = str(time_units_s) # 'seconds since {:04d}-{:02d}-{:02d}'.format(dt0.year,dt0.month,dt0.day)
        ncv.attrs['calendar'] = 'proleptic_gregorian'

        schema.write(ncin, output_file4)
