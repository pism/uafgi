import os
import subprocess
import netCDF4

def fix_output(output_file3, exception, time_units_s, output_file4):
    """Compress PISM output file while correcting time units; and removes original.
    exception:
        Exception (if any) that occurred while running PISM
    units_s: cfunits.Unit
        Units in "seconds since..." format
    """

    # Convert to NetCDF
    cmd = ['ncks', '-4', '-L', '1', '-O', output_file3, output_file4]
    subprocess.run(cmd, check=True)
    os.remove(output_file3)

    # Fix up time units to be CF compliant
    with netCDF4.Dataset(output_file4, 'a') as nc:
        nc.success = ('t' if exception is None else 'f')
        if exception is not None:
            nc.error_msg = str(exception)
        nc.variables['time'].units = str(time_units_s) # 'seconds since {:04d}-{:02d}-{:02d}'.format(dt0.year,dt0.month,dt0.day)
        nc.variables['time'].calendar = 'proleptic_gregorian'

