import numpy as np
import netCDF4
import os,subprocess
from cdo import Cdo
from uafgi import nsidc
from uafgi import giutil,cdoutil,make,ioutil

class IceRemover2(object):

    def __init__(self, bedmachine_file):
        """bedmachine-file: Local extract from global BedMachine"""
        self.bedmachine_file = bedmachine_file
        with netCDF4.Dataset(self.bedmachine_file) as nc:

            # ------- Read original thickness and bed
            self.thk = nc.variables['thickness'][:]

    def get_thk(self, termini_closed_file, index, odir, tdir):
        """Yields an ice thickness field that's been cut off at the terminus trace0
        trace0: (gline_xx,gline_yy)
            output of iter_traces()
        """

        # Select a single polygon out of the shapefile
        one_terminus = tdir.join('one_terminus_closed.shp')
        cmd = ['ogr2ogr', one_terminus, termini_closed_file, '-fid', str(index)]
        subprocess.run(cmd, check=True)

        # Cut the bedmachine file based on the shape
        cut_geometry_file = tdir.join('cut_geometry_file.nc')
        cmd = ['gdalwarp', '-cutline', one_terminus, 'NETCDF:{}:bed'.format(self.bedmachine_file), cut_geometry_file]
        subprocess.run(cmd, check=True)

        # Read the fjord mask from that file
        with netCDF4.Dataset(cut_geometry_file) as nc:
            fjord = np.logical_not(nc.variables['Band1'][:].mask)
        print('fjord sum: {} {}'.format(np.sum(np.sum(fjord)), fjord.shape[0]*fjord.shape[1]))


        # Remove downstream ice
        thk = np.zeros(self.thk.shape)
        thk[:] = self.thk[:]
        thk[fjord] = 0

        return thk

        ## Store it...
        #with netCDF4.Dataset(bedmachine_file, 'r') as nc0:
        #    with netCDF4.Dataset('x.nc', 'w') as ncout:
        #        cnc = ncutil.copy_nc(nc0, ncout)
        #        vars = list(nc0.variables.keys())
        #        cnc.define_vars(vars)
        #        for var in vars:
        #            if var not in {'thickness'}:
        #                cnc.copy_var(var)
        #        ncout.variables['thickness'][:] = thk

