import io,os
import datetime

import netCDF4
import gdal
from cdo import Cdo
import cf_units

from uafgi import iopfile,ioutil,ncutil

"""Parsers and formatters for NSIDC file sets"""


def extract_grid(in_tif, grid, out_nc, tdir):
    """Given a .tif downloaded file, extracts the grid from it as a NetCDF
    file.

    in_tif:
        .tif file to extract
    grid:
        Name of grid (implied from in_tif's filename)
    out_nc:
        .nc file to write to
    tdir:
        Temporary directory service
    """

    tmp_file = tdir.filename(suffix='.nc')

    print("Extract grid of {} to {}".format(in_tif, out_nc))
    # use gdal's python binging to convert GeoTiff to netCDF
    # advantage of GDAL: it gets the projection information right
    # disadvantage: the variable is named "Band1", lacks metadata
    ds = gdal.Open(in_tif)
    options = gdal.TranslateOptions(gdal.ParseCommandLine(
        "-co COMPRESS=DEFLATE"))
    ds = gdal.Translate(tmp_file, ds, options=options)
    ds = None

    # Copy out the netCDF file, but without Band1 var
    with netCDF4.Dataset(tmp_file) as nc:
        with netCDF4.Dataset(out_nc, 'w') as ncout:
            cnc = ncutil.copy_nc(nc,ncout)
            cnc.define_vars(['x','y','polar_stereographic'], zlib=True)
            ncout.grid = grid
            cnc.copy_data()



# -------------------------------------------------------------
class x_extract_grid(object):
    """Chooses a single TIFF file and creates a single file just
    describing the grid."""

    def __init__(self, makefile, idir, parse_fn, odir, grid):

        self.grid = grid
        self.odir = odir

        # Get list of .tif files for the given grid
        attrs = {'grid': grid, 'ext': '.tif'}
        filter_fn = iopfile.filter_attrs(attrs)
        pfiles_tif = iopfile.listdir(idir, parse_fn, filter_fn)

        # Go through just one file
        pf = pfiles_tif[0]
        self.ipath = pf.path
        self.opath = os.path.join(odir, '{}-grid.nc'.format(grid))
        self.rule = makefile.add(self.run,
            (self.ipath,),
            (self.opath,))
            
    def run(self):
        os.makedirs(os.path.split(self.opath)[0], exist_ok=True)

        with ioutil.tmp_dir(self.odir) as tdir:
            tmp_file = os.path.join(tdir, 'tmp.nc')

            print("Converting {} to {}".format(self.ipath, self.opath))
            # use gdal's python binging to convert GeoTiff to netCDF
            # advantage of GDAL: it gets the projection information right
            # disadvantage: the variable is named "Band1", lacks metadata
            ds = gdal.Open(self.ipath)
            options = gdal.TranslateOptions(gdal.ParseCommandLine(
                "-co COMPRESS=DEFLATE"))
            ds = gdal.Translate(tmp_file, ds, options=options)
            ds = None

            # Copy out the netCDF file, but without Band1 var
            with netCDF4.Dataset(tmp_file) as nc:
                with netCDF4.Dataset(self.opath, 'w') as ncout:
                    cnc = ncutil.copy_nc(nc,ncout)
                    cnc.define_vars(['x','y','polar_stereographic'], zlib=True)
                    ncout.grid = self.grid
                    cnc.copy_data()

# ---------------------------------------------------
class tiff_to_netcdf0(object):
    """Makefile macro, convert a GeoTIFF to raw NetCDF file.
    Generally used only from tiff_to_netcdf().
    pfile:
        The iopfile.PFile to convert
    odir:
        Directory to place the output"""

    def __init__(self, makefile, pfile, odir):

        self.ipath = pfile.path
        self.opath = os.path.join(odir, pfile.format(
            dir=odir,
            ext='.tiff_to_netcdf_0.nc'))
        self.rule = makefile.add(self.run,
            (self.ipath,),
            (self.opath,))

    def run(self):
        print('opath ',self.opath)
        os.makedirs(os.path.split(self.opath)[0], exist_ok=True)

        print("Converting {} to {}".format(self.ipath, self.opath))
        # use gdal's python binging to convert GeoTiff to netCDF
        # advantage of GDAL: it gets the projection information right
        # disadvantage: the variable is named "Band1", lacks metadata
        ds = gdal.Open(self.ipath)
        ds = gdal.Translate(self.opath, ds)
        ds = None
# -------------------------------------------------------------------------
class tiff_to_netcdf(object):
    """Makefile macro, convert a GeoTIFF to NetCDF file with fixed metadata.
    pfile:
        The iopfile.PFile to convert
    odir:
        Directory to place the output
    reftime:    
        Reference time for CF Times in the file
    """

    def __init__(self, makefile, pfile, odir, reftime='2008-01-01'):

        sub = tiff_to_netcdf0(makefile, pfile, odir)

        # This deduces the mid-point (nominal) date from the filename
        self.time_bounds = (pfile['startdate'], pfile['enddate'])
        self.nominal_date = pfile['startdate'] + (pfile['enddate'] - pfile['startdate']) / 2
        self.reftime = reftime
        self.parameter = pfile['parameter']
        self.ipath = sub.opath
        self.opath = os.path.join(odir, pfile.format(
            dir=odir,
            ext='.nc'))

        self.rule = makefile.add(self.run, (self.ipath,), (self.opath,))

    def run(self):
        cdo = Cdo()

        # Set the time axis
        inputs = [
            '-setreftime,{}'.format(self.reftime),  # Somehow reftime is being ignored
            '-setattribute,{}@units="m year-1"'.format(self.parameter),
            '-chname,Band1,{}'.format(self.parameter),
            '{}'.format(self.ipath)]
        cdo.settaxis(
            self.nominal_date.isoformat(),
            input=' '.join(inputs),
            output=self.opath,
            options="-f nc4 -z zip_2")

        # Add time bounds --- to be picked up by cdo.mergetime
        # https://code.mpimet.mpg.de/boards/2/topics/1115
        with netCDF4.Dataset(self.rule.outputs[0], 'a') as nc:
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
            for ix,time in enumerate(self.time_bounds):
                tbv[0,ix] = cfu.date2num(time)



# -------------------------------------------------------------------------
class tiffs_to_netcdfs(object):
    """Makefile macro, convert a directory full of GeoTIFFs to NetCDF.
    idir:
        Input directory of GeoTIFF files
    parse_fn:
        Parses filenames of GeoTIFF files
    odir:
        Directory for output files
    filter_attrs:
        Files converted must match these attributes.
        For example: source, grid, parameter
    reftime:    
        Reference time for CF Times in the file
    blacklist:
        Files to NOT include
    max_files:
        Maximum number of files to include
    """

    def __init__(
        self, makefile, idir, parse_fn, odir,
        reftime='2008-01-01', blacklist=set(), max_files=10000000,
        filter_attrs=dict()):

        self.blacklist = blacklist

        # Get list of .tif files
        attrs = dict(filter_attrs.items())
        attrs['ext'] = '.tif'
        filter_fn = iopfile.filter_attrs(attrs)
        self.pfiles_tif = iopfile.listdir(idir, parse_fn, filter_fn)

        # Go through each file
        inputs = list()
        outputs = list()
        for ix,pf in enumerate(self.pfiles_tif):

            # Don't iterate over blacklisted items
            if pf.leaf in self.blacklist:
                continue

            # Cut it off early
            if ix >= max_files:
                break

            rule = tiff_to_netcdf(makefile, pf, odir, reftime=reftime).rule
            inputs += rule.inputs
            outputs += rule.outputs

        self.rule = makefile.add(None, inputs, outputs)
