import numpy as np
from uafgi import ioutil,make,cdoutil,ncutil
from cdo import Cdo
import os
import datetime
import subprocess
import netCDF4
import re

def process_year(ifname, time_bounds, grid_file, ofname, tdir):
    """Extracts a local region from a single-year ItsLive file.
    Does some additional fixups as well.

    ifname:
        Name of original single-year ItsLive file
    year: int
        Year the original file is for
    grid_file:
        NetCDF file defining the destination grid
    ofname:
        Name of file to write
    tdir: ioutil.TmpDir
        Temporary directory
    """

    # Remove non-date portion of time_bounds
    # (CDO doesn't like it on -setreftime)
    time_bounds = [x if isinstance(x,datetime.date) else x.date() for x in time_bounds]

    cdo = Cdo()
    odir = os.path.split(ofname)[0]

    # Extract each variable separately
    merge_files = list()
    for vname in ('VX', 'VY'):

        # Use GDAL to extract just local region
        tmp_v1 = tdir.filename() + '.nc'
        cdoutil.extract_region_onevar(ifname, grid_file, vname, tmp_v1)

        # Add time
        tmp_v2 = tdir.filename() + '.nc'

#        time_bounds = (datetime.datetime(year,1,1), datetime.datetime(year,12,31))
#        reftime = datetime.date(year,1,1)
        reftime = (time_bounds[0] + (time_bounds[1] - time_bounds[0]) / 2).date()
        cdoutil.set_time_axis(tmp_v1, tmp_v2, time_bounds, reftime)


        # Fix units
        with netCDF4.Dataset(tmp_v2, 'a') as nc:
            ncv = nc.variables[vname]

            # Fix units
            if ncv.units == 'm/y':
                ncv.units = 'm year-1'

            # Eliminate missing values
            # See PISM:
            # commit 60739ae417da1c4b106d9a4d9ab22a165194bdb6
            # Author: Constantine Khrulev <ckhroulev@alaska.edu>
            # Date:   Wed Oct 14 11:21:18 2020 -0800
            #     Stop PISM if a variable in an input file has missing values    
            #     We check if some values of a variable match the _FillValue attribute.
            val = ncv[:].data    # NetCDF reads a masked array
            print('FillValueX ',ncv._FillValue, np.sum(np.sum(val==ncv._FillValue)))
            val[val==ncv._FillValue] = 0
            ncv[:] = val

        # Add to files to merge
        merge_files.append(tmp_v2)

    # Simple merge
    tmp4 = tdir.opath(ofname, '_4.nc')
    print('******** Merging {} -> {}'.format(merge_files, ofname))
    cdoutil.merge(cdo.merge, merge_files, tmp4, tdir)

    # Rename variables to what PISM expects
    # HINT: If presence is intended to be optional, then prefix
    # old variable name with the period character '.', i.e.,
    # 'ncrename -v .vy,v_ssa_bc'. With this syntax ncrename would
    # succeed even when no such variable is in the file.
    # tmp_v3 = make.opath(ofname, tdir, '_{}2.nc'.format(vname))
    tmp_v3 = tdir.opath(ofname, '_{}2.nc'.format(vname))
    cmd = ['ncrename', '-O', '-v', '.VX,u_ssa_bc', '-v', '.VY,v_ssa_bc',
        tmp4, ofname]
    subprocess.run(cmd, check=True)

 
#fileRE = re.compile(r'vel_(\d\d\d\d)_(\d\d)_(\d\d)_(\d\d\d\d)_(\d\d)_(\d\d).nc')
def process_years(year_files, time_bndss, grid, grid_file, allyear_file, tdir):
    """Process multiple 1-year global Its-Live files into a single
    multi-year local file."""

    cdo = Cdo()

    odir = os.path.split(allyear_file)[0]
    oyear_files = list()

    for ifname,time_bnds in zip(year_files,time_bndss):
        ofname = tdir.filename()
        print('ofname = {}'.format(ofname))
        process_year(ifname, time_bnds, grid_file, ofname, tdir)
        oyear_files.append(ofname)

    cdoutil.merge(cdo.mergetime, oyear_files, allyear_file, tdir,
        options='-f nc4 -z zip_2')

    # Add the grid as an attribute
    with netCDF4.Dataset(allyear_file, 'a') as nc:
        nc.grid = grid
        nc.grid_file = grid_file


def merge_to_pism_rule(grid, grid_file, ifpattern, range_dt0, range_dt1, odir):
    """Merges a bunch of Its-Live files while extracting a local region from them.

    grid:
        Name of the local grid to which data are to be regridded
    grid_file:
        Sample NetCDF file with the details of grid
    ifpattern:
        Eg, 'data/itslive/GRE_G0240_{}.nc' where {}=year
    years: iterable(int)
        Years to process
    """

    # ---------- Prepare the rule
    fileRE = re.compile(r'vel_(\d\d\d\d)-(\d\d)-(\d\d)_(\d\d\d\d)-(\d\d)-(\d\d).nc')
    iyear_filesx = []
    ifdir = os.path.split(ifpattern)[0]
    for leaf in os.listdir(ifdir):
        match = fileRE.match(leaf)
        if match is None:
            continue

        dt0 = datetime.datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)))
        dt1 = datetime.datetime(
            int(match.group(1))+1,
            int(match.group(2)),
            int(match.group(3)))


#        print([match.group(i) for i in range(1,7)])

        # The filename has illegal date: yyyy-06-31
        #dt1 = datetime.datetime(
        #    int(match.group(4)),
        #    int(match.group(5)),
        #    int(match.group(6))) \
        #    + datetime.timedelta(days=1)

        # Make sure it's in range
        if (dt1 <= range_dt0) or (dt0 >= range_dt1):
            continue

        iyear_filesx.append(( (dt0,dt1), os.path.join(ifdir,leaf) ))

    iyear_filesx.sort()
#    iyear_filesx = iyear_filesx[:2]    # TODO: Debugging

    time_bndss =  [x[0] for x in iyear_filesx]
    iyear_files = [x[1] for x in iyear_filesx]

    allyear_file = make.opath(
        ifpattern.format('{}_{:04}_{:04}'.format(grid, range_dt0.year, range_dt1.year)),
        odir, '')

    def action(tdir):
        process_years(iyear_files, time_bndss, grid, grid_file,
            allyear_file, tdir)

    return make.Rule(action,
        iyear_files + [grid_file], (allyear_file,))

# ---------------------------------------------------------------------
