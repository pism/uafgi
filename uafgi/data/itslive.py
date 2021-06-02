import numpy as np
from uafgi import ioutil,make,cdoutil,ncutil
from cdo import Cdo
import os
import datetime
import subprocess
import netCDF4
import re
import uafgi.data

def process_year(ifname, VXY, time_bounds, grid_file, ofname, tdir):
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
    VX,VY = VXY
    zerofill = False
    for vname in VXY:    # ('VX', 'VY')

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
            fv = ncv._FillValue
            if fv == 0:
                fv = -32767

            # Change FillValue -> 0
            val = ncv[:].data    # NetCDF reads a masked array
            val[val==fv] = 0
            ncv[:] = val

        # Add to files to merge
        merge_files.append(tmp_v2)

    # Simple merge
    tmp4 = tdir.opath(ofname, '_4.nc')
    print('******** Merging {} -> {}'.format(merge_files, ofname))
    cdoutil.merge(cdo.merge, merge_files, tmp4, tdir)

    # Rename variables to what PISM expects; while "filling" missing values with 0
    # HINT: If presence is intended to be optional, then prefix
    # old variable name with the period character '.', i.e.,
    # 'ncrename -v .vy,v_ssa_bc'. With this syntax ncrename would
    # succeed even when no such variable is in the file.
    # tmp_v3 = make.opath(ofname, tdir, '_{}2.nc'.format(vname))
    ovnames = ('u_ssa_bc', 'v_ssa_bc')
    with netCDF4.Dataset(tmp4) as ncin:
        schema = ncutil.Schema(ncin)
        for ivname,ovname in zip(VXY,ovnames):
            sv = schema.rename(ivname, ovname)
            for attr in ('_FillValue', 'missing_value'):
                if attr in sv.attrs:
                    del sv.attrs[attr]

        with netCDF4.Dataset(ofname, 'w') as ncout:
            schema.create(ncout)

            for ovname in ovnames:
                del schema.vars[ovname]
            schema.copy(ncin, ncout)

            for ivname,ovname in zip(VXY,ovnames):
                ncv = ncin.variables[ivname]
                fv = ncv._FillValue
                if fv == 0:
                    fv = -32767

                val = ncv[:].data
                ncout.variables[ovname][:] = val
 
def process_years(year_files, VXY, time_bndss, grid, grid_file, allyear_file, tdir):
    """Process multiple 1-year global Its-Live files into a single
    multi-year local file."""

    cdo = Cdo()

    odir = os.path.split(allyear_file)[0]
    oyear_files = list()

    for ifname,time_bnds in zip(year_files,time_bndss):
        ofname = tdir.filename()
        print('ofname = {}'.format(ofname))
        process_year(ifname, VXY, time_bnds, grid_file, ofname, tdir)
        oyear_files.append(ofname)

    cdoutil.merge(cdo.mergetime, oyear_files, allyear_file, tdir,
        options='-f nc4 -z zip_2')

    # Add the grid as an attribute
    with netCDF4.Dataset(allyear_file, 'a') as nc:
        nc.grid = grid
        nc.grid_file = grid_file



class W21Merger:
	"""Input for merge_to_pism_rule() to process Wood 2021 velocity files."""
    RE = re.compile(r'vel_(\d\d\d\d)-(\d\d)-(\d\d)_(\d\d\d\d)-(\d\d)-(\d\d)\.nc')
    VXY = ('VX', 'VY')
    idir = uafgi.data.join('wood2021', 'velocities')

    @staticmethod
    def parse(leaf):
        match = W21Merger.RE.match(leaf)
        if match is None:
            return None

        dt0 = datetime.datetime(
            int(match.group(1)),
            int(match.group(2)),
            int(match.group(3)))
        dt1 = datetime.datetime(
            int(match.group(1))+1,
            int(match.group(2)),
            int(match.group(3)))

        return dt0,dt1

    @staticmethod
    def ofname(grid, dt0, dt1):
        return uafgi.data.join_outputs(
            'wood2021', 'velocities',
            'vel_{}_{:04}_{:04}.nc'.format(grid, dt0, dt1))


class ItsliveMerger:
	"""Snap-in for merge_to_pism_rule() to process Its LIVE velocity files."""

    # What the input data filenames look like
    RE = re.compile(r'GRE_G0240_(\d\d\d\d)\.nc')
    # Name of ice velocities variables in the data files
    VXY = ('vx', 'vy')
    # Where the foind the input data files
    idir = uafgi.data.join('itslive')

    @staticmethod
    def parse(leaf):
        """Parse a date range out of a filename; or discard (None) as not part
        of this dataset."""
        match = ItsliveMerger.RE.match(leaf)
        if match is None:
            return None

        dt0 = datetime.datetime(int(match.group(1)), 1,1)    # Jan 1, <year>
        dt1 = datetime.datetime(int(match.group(1))+1, 1,1)    # Jan 1, <next year>

        return dt0,dt1

    @staticmethod
    def ofname(grid, year0, year1):
        """The output filename"""
        return uafgi.data.join_outputs(
            'itslive', 'GRE_G0240_{}_{}_{}.nc'.format(grid, year0, year1-1))



def merge_to_pism_rule(grid, Merger, range_dt0, range_dt1):
    """Merges a bunch of Its-Live or Wood 2021 velocity files while extracting a local region from them.
    Produces one multi-year file, selected for the given grid.

    grid:
        Name of the local grid to which data are to be regridded
    Merger:
        Either W21Merger (Wood 2021 velocity files) or ItsliveMerger (Its LIVE velocity files)
        ...depending on which kind of file is being processed.
    range_dt0, range_dt1:
        Date range within which to process input files.
    """

    grid_file = uafgi.data.measures_grid_file(grid)

    # ---------- Prepare the rule
    iyear_filesx = []
    for leaf in os.listdir(Merger.idir):
        parse = Merger.parse(leaf)
        if parse is None:
            continue
        dt0,dt1 = parse

        # Make sure it's in range
        if (dt1 <= range_dt0) or (dt0 >= range_dt1):
            continue

        iyear_filesx.append(( (dt0,dt1), os.path.join(Merger.idir,leaf) ))

    iyear_filesx.sort()

    time_bndss =  [x[0] for x in iyear_filesx]
    iyear_files = [x[1] for x in iyear_filesx]

    allyear_file = Merger.ofname(grid, range_dt0.year, range_dt1.year)

    def action(tdir):
        process_years(iyear_files, Merger.VXY, time_bndss, grid, grid_file,
            allyear_file, tdir)

    return make.Rule(action,
        iyear_files + [grid_file], (allyear_file,))

# ---------------------------------------------------------------------
