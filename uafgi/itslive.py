from uafgi import ioutil,make,cdoutil
from cdo import Cdo
import os
import datetime
import subprocess
import netCDF4

def process_year(ifname, year, grid_file, ofname, tdir):
    """Transforms an Its-Live file from original format to one ready to
    merge."""

    cdo = Cdo()
    odir = os.path.split(ofname)[0]

    # Extract each variable separately
    merge_files = list()
    for vname in ('vx', 'vy', 'v'):

        # Use GDAL to extract just local region
        #tmp_v1 = make.opath(ofname, tdir, '_{}1.nc'.format(vname))
        #tmp_v1 = tdir.opath(ofname, '_{}1.nc'.format(vname))
        tmp_v1 = tdir.filename() + '.nc'
        print('tmp_v1 = {}'.format(tmp_v1))
        cdoutil.extract_region(ifname, grid_file, vname, tmp_v1)

        # Add time
        #tmp_v2 = make.opath(ofname, tdir, '_{}2.nc'.format(vname))
        tmp_v2 = tdir.filename() + '.nc'
        time_bounds = (datetime.datetime(year,1,1), datetime.datetime(year,12,31))
        reftime = datetime.date(year,1,1)
        cdoutil.set_time_axis(tmp_v1, tmp_v2, time_bounds, reftime)


        # Fix units
        with netCDF4.Dataset(tmp_v2, 'a') as nc:
            ncv = nc.variables[vname]
            if ncv.units == 'm/y':
                ncv.units = 'm year-1'

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
    cmd = ['ncrename', '-O', '-v', '.vx,u_ssa_bc', '-v', '.vy,v_ssa_bc',
        tmp4, ofname]
    subprocess.run(cmd, check=True)


def process_years(year_files, years, grid_file, allyear_file, tdir):
    """Process multiple 1-year Its-Live files into a single file."""
    cdo = Cdo()

    odir = os.path.split(allyear_file)[0]
    oyear_files = list()

    for ifname,year in zip(year_files,years):
        ofname = tdir.join('year_{:04}.nc'.format(year))
        print('ofname = {}'.format(ofname))
        process_year(ifname, year, grid_file, ofname, tdir)
        oyear_files.append(ofname)

    cdoutil.merge(cdo.mergetime, oyear_files, allyear_file, tdir,
        options='-f nc4 -z zip_2')


def merge_to_pism_rule(grid, grid_file, ifpattern, years, odir):
    """Merges a bunch of Its-Live files

    ifpattern:
        Eg, 'data/itslive/GRE_G0240_{}.nc'
    years: iterable(int)
        Years to process
    """

    # ---------- Prepare the rule
    iyear_files = [
        ifpattern.format('{:04}'.format(year))
        for year in years]
    allyear_file = make.opath(
        ifpattern.format('{}_{:04}_{:04}'.format(grid, years[0],years[-1])),
        odir, '')

    def action(tdir):

        process_years(iyear_files, years, grid_file,
            allyear_file, tdir)

    return make.Rule(action,
        iyear_files, (allyear_file,))
