from cdo import Cdo
import os.path
from uafgi import ioutil
import netCDF4
import cf_units
from uafgi import functional

"""Utilities for working with the Python CDO interface"""
def _large_merge(cdo_merge_operator, input, output, tmp_files, max_merge=30, **kwargs):
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
        try:
            chunks = [input[x:x+max_merge] for x in range(0, len(input), max_merge)]
            for chunk in chunks:
                ochunk = next(tmp_files)
                input1.append(ochunk)
                _large_merge(cdo_merge_operator, chunk, ochunk, tmp_files, max_merge, **kwargs)

            _large_merge(cdo_merge_operator, input1, output, tmp_files, max_merge, **kwargs)
        finally:
            # Remove our temporary files
            for path in input1:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
    else:
#        print('CDO Merge')
#        print('INPUT ',input)
#        print('OUTPUT ',output)
        cdo_merge_operator(input=input, output=output, **kwargs)


def merge(cdo_merge_operator, inputs, output, max_merge=30, **kwargs):
    """Recursively merge large numbers of files using a CDO merge-type operator.
    Also appropriate for "small" merges.

    cdo_merge_operator:
        The CDO operator used to merge; Eg: cdo.mergetime
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
    with ioutil.TmpFiles(os.path.join(odir, 'tmp')) as tmp_files:
        _large_merge(cdo_merge_operator, inputs, output, tmp_files, max_merge=max_merge, **kwargs)

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
@functional.memoize
class FileBounds(object):
    """Reads spatial extents from NetCDF file.
    May be used, eg, as:
                '-projwin', str(x0), str(y1), str(x1), str(y0),
                '-tr', str(dx), str(dy),
    Returns:
        self.x0, self.x1:
            Min, max of region in the file
        self.dx:
            Grid spacing in x direction
        welf.wks_s:
            Coordinate reference system (CRS) used in the file
    """
    def __init__(self, grid_file):
        """Obtains bounding box of a grid.
        Returns: x0,x1,y0,y1
        """

        with netCDF4.Dataset(grid_file) as nc:
            xx = nc.variables['x'][:]
            self.nx = len(xx)
            self.dx = xx[1]-xx[0]
            half_dx = .5 * self.dx
            self.x0 = round(xx[0] - half_dx)
            self.x1 = round(xx[-1] + half_dx)

            yy = nc.variables['y'][:]
            self.ny = len(yy)
            self.dy = yy[1]-yy[0]
            half_dy = .5 * self.dy
            self.y0 = round(yy[0] - half_dy)
            self.y1 = round(yy[-1] + half_dy)

            self.crs = nc.variables['polar_stereographic'].spatial_ref

            ncv = nc.variables['polar_stereographic']
            if hasattr(ncv, 'GeoTransform'):
                sgeotransform = ncv.GeoTransform
                self.geotransform = tuple(float(x) for x in sgeotransform.split(' ') if len(x) > 0)


# --------------------------------------------------------
def extract_region(ifname, grid_file, vname, ofname):
    """ifname:
        Name of input file from which to extract the region
    grid_file:
        Name of file containing x and y grid coordinate values
    vname:
        Name of (single) variable to extract
    ofname:
        Name of output files
    """

    fb = FileBounds(grid_file)

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
