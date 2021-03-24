import traceback
import os
import numpy as np
import pandas as pd
import datetime
from uafgi import argutil,gdalutil,glacier,bedmachine
import uafgi.data
from uafgi.pism import pismutil
from uafgi.pism import calving0
import netCDF4
import PISM

blackout_types = {
    ('shapely.geometry.multipoint', 'MultiPoint'),
    ('shapely.geometry.polygon', 'Polygon'),
#    ('shapely.geometry.point', 'Point'),
    ('shapely.geometry.linestring', 'LineString'),
    ('numpy', 'ndarray'),
}



def run_pism(ns481_grid, fjord_classes, velocity_file, year, output_file, tdir, dry_run=False, row=None, **pism_kwargs0):
    """Does a single PISM run
    ns481_grid:
        Name of the grid on which this runs (as per NSIDC-0481 dataset)
    fjord_classes:
        Classification of fjord areas, as initial condition for run
    bedmachine_file: <filename>
        Local bedmachine file extract
    velocity_file: <filename>
        File of velocities; must have same CRS and bounds as bedmachine_file
    ofiles_only:
        True if this should just compute output filenames and return
        (for use in make rules)
    dry_run:
        If true, just return (inputs, outputs)
    row: pd.Series
        Row of a Pandas Dataframe, to write to output file
    pism_kwargs0:
        kwargs given to PISM run
    """

    # ============ Determine input/output filenames

    # Determine the local grid
    grid = ns481_grid
    grid_file = uafgi.data.measures_grid_file(grid)
    grid_info = gdalutil.FileInfo(grid_file)

    bedmachine_file0 = uafgi.data.bedmachine_local(grid)


    if dry_run:
        inputs = [grid_file, bedmachine_file0, velocity_file]
        outputs = [output_file]
        return inputs, outputs

    # ============================================


    # Get total kwargs to use for PISM
    default_kwargs = dict(calving0.FrontEvolution.default_kwargs.items())
    default_kwargs['min_ice_thickness'] = 50.0    # See TODO below
    kwargs = argutil.select_kwargs(pism_kwargs0, default_kwargs)


    # Clear away ice below the terminus

    # Get ice thickness
    with netCDF4.Dataset(bedmachine_file0) as nc:
        thickness = nc.variables['thickness'][:]

    # Remove ice downstream of the terminus
    down_fjord = np.isin(fjord_classes, glacier.LT_TERMINUS)
    thickness[down_fjord] = 0

    # Copy original local BedMachine file, with new ice terminus
    bedmachine_file1 = tdir.filename(suffix='.nc')
    bedmachine.replace_thk(bedmachine_file0, bedmachine_file1, thickness)

    # Obtain start and end time in PISM units (seconds)
    fb = gdalutil.FileInfo(velocity_file)
    years_ix = dict((dt.year,ix) for ix,dt in enumerate(fb.datetimes))
    itime = years_ix[year]    # Index in velocity file

    dt0 = datetime.datetime(year,1,1)
    t0_s = fb.time_units_s.date2num(dt0)
    #dt1 = datetime.datetime(year+1,1,1)
    dt1 = datetime.datetime(year,4,1)
    t1_s = fb.time_units_s.date2num(dt1)

    # ---------------------------------------------------------------

    print('============ Running year {}'.format(year))
    output_file3 = tdir.filename()
    print('     ---> {}'.format(output_file3))

    # Prepare to store the output file
#    odir = os.path.split(output_file)[0]
#    if len(odir) > 0:
#        os.makedirs(odir, exist_ok=True)

    try:

        # The append_time=True argument of prepare_output
        # determines if after this call the file will contain
        # zero (append_time=False) or one (append_time=True)
        # records.
        output = PISM.util.prepare_output(output_file3, append_time=False)

        # TODO: Add a time_units and calendar argument to prepare_output()
        # https://github.com/pism/pism/commit/1cd1719189f1155bf56b4488338f1d6e53c29659

        #### I need to mimic this: Ross_combined.nc plus the script that made it
        # Script in the main PISM repo, it's in examples/ross/preprocess.py
        # bedmachine_file = "~/github/pism/pism/examples/ross/Ross_combined.nc"
        # bedmachine_file = "Ross_combined.nc"
        ctx = PISM.Context()
        # TODO: Shouldn't this go in calving0.init_geometry()?
        ctx.config.set_number("geometry.ice_free_thickness_standard", kwargs['min_ice_thickness'])

        grid = calving0.create_grid(ctx.ctx, bedmachine_file1, "thickness")
        geometry = calving0.init_geometry(grid, bedmachine_file1, kwargs['min_ice_thickness'])

        ice_velocity = calving0.init_velocity(grid, velocity_file)
        print('ice_velocity sum: '.format(np.sum(np.sum(ice_velocity))))

        # NB: For debugging I might use a low value of sigma_max to make SURE things retreat
        # default_kwargs = dict(
        #     ice_softness=3.1689e-24, sigma_max=1e6, max_ice_speed=5e-4)
#        fe_kwargs = dict(sigma_max=0.1e6)
        front_evolution = calving0.FrontEvolution(grid, sigma_max=kwargs['sigma_max'])

        # ========== ************ DEBUGGING *****************
        #xout = PISM.util.prepare_output('x.nc', append_time=False)
        #PISM.append_time(xout, front_evolution.config, 17)
        #geometry.ice_thickness.write(xout)
        #geometry.cell_type.write(xout)


        # Iterate through portions of (dt0,dt1) with constant velocities
        ice_velocity.read(velocity_file, itime)   # 0 ==> first record of that file (if time-dependent)
        front_evolution(geometry, ice_velocity,
           t0_s, t1_s,
           output=output)
        exception = None
    except Exception as e:
        print('********** Error: {}'.format(str(e)))
        traceback.print_exc() 
        exception = e
    finally:
        output.close()

    # ----------------------- Post-processing
    output_file_tmp = tdir.filename()
    pismutil.fix_output(output_file3, exception, fb.time_units_s, output_file_tmp)

    with netCDF4.Dataset(output_file_tmp, 'a') as nc:
        # Debug
        ncv = nc.createVariable('fjord_classes', 'i1', ('y','x'), zlib=True)
        ncv[:] = fjord_classes

        # Add parameter info
        nc.creator = '04_run_experiment.py'
        nc.ns481_grid = ns481_grid
        nc.velocity_file = velocity_file
        nc.year = year
        for key,val in pism_kwargs0.items():
            nc.setncattr(key,val)

        # Add info from Pandas Series
        for col,val in row.items():
            mt = (type(val).__module__, type(val).__name__)
            if mt not in blackout_types:
                nc.setncattr(col,val)

    # ------------------- Create final output file
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)
    os.rename(output_file_tmp, output_file)
