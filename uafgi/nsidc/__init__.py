from uafgi import iopfile
import io,os
import datetime
import gdal
from cdo import Cdo

"""Parsers and formatters for NSIDC file sets"""


# -------------------------------------------------------------
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
        parameter = self.parameter
        reftime = self.reftime
        inputs = [
            '-setreftime,{}'.format(self.reftime),
            '-setattribute,{}@units="m year-1"'.format(self.parameter),
            '-chname,Band1,{}'.format(self.parameter),
            '{}'.format(self.ipath)]
        cdo.settaxis(
            self.nominal_date.isoformat(),
            input=' '.join(inputs),
            output=self.opath,
            options="-f nc4 -z zip_2")
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
