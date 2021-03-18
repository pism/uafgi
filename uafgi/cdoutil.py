from cdo import Cdo
import os.path
from uafgi import ioutil
import netCDF4
import cf_units
import subprocess
from uafgi import functional,cfutil,gdalutil

"""Utilities for working with the Python CDO interface"""
def _large_merge(cdo_merge_operator, input, output, tdir, max_merge=30, **kwargs):
    """
    max_merge:
        Maximum number of files to merge in a single CDO command
    kwargs:
        Additional arguments given to the CDO operator.
        Eg for cdo.merge(): options='-f nc4'
        https://code.mpimet.mpg.de/projects/cdo/wiki/Cdo%7Brbpy%7D#onlineoffline-help
    """
    print('_large_merge', len(input), output)
    odir = os.path.split(output)[0]

    if len(input) > max_merge:

        input1 = list()
        with tdir.subdir() as tdir1:
            chunks = [input[x:x+max_merge] for x in range(0, len(input), max_merge)]
            for chunk in chunks:
                ochunk = tdir.filename()
                input1.append(ochunk)
                _large_merge(cdo_merge_operator, chunk, ochunk, tdir1, max_merge, **kwargs)

            _large_merge(cdo_merge_operator, input1, output, tdir1, max_merge, **kwargs)

    else:
#        print('CDO Merge')
#        print('INPUT ',input)
#        print('OUTPUT ',output)
        cdo_merge_operator(input=input, output=output, **kwargs)


def merge(cdo_merge_operator, inputs, output, tdir, max_merge=30, **kwargs):
    """Recursively merge large numbers of files using a CDO merge-type operator.
    Also appropriate for "small" merges.

    cdo_merge_operator:
        The CDO operator used to merge; Eg: cdo.mergetime, cdo.merge
    inputs:
        Names of the input files to merge
    output:
        The output files to merge
    kwargs:
        Additional arguments to supply to the CDO merge command
    max_merge:
        Maximum number of files to merge in a single CDO command.
        This cannot be too large, lest it overflow the number of available OS filehandles.
    """

    print('Merging {} files into {}'.format(len(inputs), output))
    odir = os.path.split(output)[0]
    _large_merge(cdo_merge_operator, inputs, output, tdir, max_merge=max_merge, **kwargs)

# -------------------------------------------------------------
def set_time_axis(ifname, ofname, time_bounds, reftime):
    """Adds time to a NetCDF file; allows for later use with cdo.merge"""
    cdo = Cdo()


    # Set the time axis
    inputs = [
        '-setreftime,{}'.format(reftime),  # Somehow reftime is being ignored
        '{}'.format(ifname)]
    nominal_date = time_bounds[0] + (time_bounds[1] - time_bounds[0]) / 2
    cdo.settaxis(
        nominal_date.isoformat(),
        input=' '.join(inputs),
        output=ofname,
        options="-f nc4 -z zip_2")

    # Add time bounds --- to be picked up by cdo.mergetime
    # https://code.mpimet.mpg.de/boards/2/topics/1115
    with netCDF4.Dataset(ofname, 'a') as nc:
        nctime = nc.variables['time']
        timeattrs = [(name,nctime.getncattr(name)) for name in nctime.ncattrs()]
        nc.variables['time'].bounds = 'time_bnds'

        nc.createDimension('bnds', 2)
        tbv = nc.createVariable('time_bnds', 'd', ('time', 'bnds',))
        # These attrs don't end up in the final merged time_bnds
        # But they do seem to be important to keep the final value correct.
        for name,val in timeattrs:
            tbv.setncattr(name, val)

        cfu = cf_units.Unit(nctime.units, nctime.calendar)
        for ix,time in enumerate(time_bounds):
            tbv[0,ix] = cfu.date2num(time)


# --------------------------------------------------
def compress(ipath, opath):
    """Compress a NetCDF file"""
    cmd = ['ncks', '-4', '-L', '1', ipath, opath]
    subprocess.run(cmd, check=True)
# --------------------------------------------------

# --------------------------------------------------------
def extract_region_onevar(ifname, grid_file, vname, ofname):
    """Extracts a local region from a larger file (eg BedMachine)
    ifname:
        Name of input file from which to extract the region
    grid_file:
        Name of file containing x and y grid coordinate values
    vname:
        Name of (single) variable to extract
    ofname:
        Name of output files
    """

    fb = gdalutil.FileInfo(grid_file)

    # For Jakobshavn: gdal_translate
    #    -r average -projwin -219850 -2243450 -132050 -2318350 
    #    -tr 100 100
    #    NETCDF:outputs/bedmachine/BedMachineGreenland-2017-09-20_pism.nc4:thickness
    #    ./out4.nc

    cmd = ['gdal_translate',
        '-r', 'average',
        '-projwin', str(fb.x0), str(fb.y1), str(fb.x1), str(fb.y0),
        '-tr', str(fb.dx), str(fb.dy),
        'NETCDF:' + ifname + ':'+vname,
        ofname]

    #print('*********** ', ' '.join(cmd))
    subprocess.run(cmd, check=True)
# -------------------------------------------------
def extract_region(ifname, grid_file, vnames, ofname, tdir):
    """Extracts a local region from a larger file (eg BedMachine)
    ifname:
        Name of input file from which to extract the region
    grid_file:
        Name of file containing x and y grid coordinate values
    vnames:
        List of variables to extract
    ofname:
        Name of output files
    """
    subs = list()
    with tdir.subdir() as tdir1:
        for vname in vnames:
            ofname1 = tdir1.join(vname+'.nc')
            subs.append(ofname1)
            extract_region_onevar(ifname, grid_file, vname, ofname1)

        cdo = Cdo()
        merge(cdo.merge, subs, ofname, tdir1)


