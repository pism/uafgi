import traceback
import os,sys
import numpy as np
import pandas as pd
import datetime
from uafgi import argutil,gdalutil,glacier,bedmachine,ncutil,make
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
    ('uafgi.gdalutil', 'FileInfo'),
}

blackout_names = {
    'ns642_years', 'ns642_termini'
}



def run_pism(ns481_grid, fjord_classes, velocity_file, output_file, tdir, dry_run=False, attrs=dict(),
    itime=None, year=None,
    dt_s=90*86400., remove_downstream_ice=True, delete_vars=list(), **pism_kwargs0):
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
    attrs: pd.Series (or just a dict)
        Row of a Pandas Dataframe, to write to output file
    itime: int
        Index into time dimension of velocity file
        NOTE: Either set year or itime
    year: int
        Year (of time dimension) to use
        NOTE: Either set year or itime
    dt_s: [s]
        Length of time to run for
    remove_downstream_ice:
        Should ice downstream of the terminus line be removed?
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
        outputs = [output_file_raw]
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
    if remove_downstream_ice:
        down_fjord = np.isin(fjord_classes, glacier.LT_TERMINUS)
        thickness[down_fjord] = 0

    # Copy original local BedMachine file, with new ice terminus
    bedmachine_file1 = tdir.filename(suffix='.nc')
    bedmachine.replace_thk(bedmachine_file0, bedmachine_file1, thickness)

    # Obtain start and end time in PISM units (seconds)
    fb = gdalutil.FileInfo(velocity_file)
    if itime is None:
        years_ix = dict((dt.year,ix) for ix,dt in enumerate(fb.datetimes))
        itime = years_ix[year]    # Index in velocity file
    elif year is None:
        year = fb.datetimes[itime].year

    dt0 = datetime.datetime(year,1,1)
    t0_s = fb.time_units_s.date2num(dt0)
    t1_s = t0_s + dt_s
    # ---------------------------------------------------------------

    print('============ Running year {}'.format(year))
    output_file3 = tdir.filename()
    print('     ---> {}'.format(output_file3))
    sys.stdout.flush()

    # Prepare to store the output file
#    odir = os.path.split(output_file)[0]
#    if len(odir) > 0:
#        os.makedirs(odir, exist_ok=True)

    try:

        # The append_time=True argument of prepare_output
        # determines if after this call the file will contain
        # zero (append_time=False) or one (append_time=True)
        # records.
        sys.stdout.flush()
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
        sys.stdout.flush()
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
    pismutil.fix_output(output_file3, exception, fb.time_units_s, output_file_tmp, delete_vars=delete_vars)

    with netCDF4.Dataset(output_file_tmp, 'a') as nc:

        # Debug
        if fjord_classes is not None:
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
        for col,val in attrs.items():
            mt = (type(val).__module__, type(val).__name__)
            if mt not in blackout_types and col not in blackout_names:
                try:
                    nc.setncattr(col,val)
                except:
                    print('Error pickling column {} = {}'.format(col,val))
                    raise

    # ------------------- Create final output file
    os.makedirs(os.path.split(output_file)[0], exist_ok=True)
    os.rename(output_file_tmp, output_file)




def get_von_Mises_stress(ns481_grid, velocity_file, output_file, tdir, **kwargs):
    """Runs PISM just a baby amount, to get the von Mises stress on the first timestep."""
    args = (ns481_grid, None, velocity_file, output_file, tdir)
    kwargs2 = kwargs.copy()
    kwargs2['remove_downstream_ice'] = False    # Don't cut off any ice below the terminus
    kwargs2['dt_s'] = 100.                     # Run only briefly
    kwargs2['delete_vars'] = {'ice_area_specific_volume', 'thk', 'total_retreat_rate', 'flux_divergence'}

    return run_pism(*args, **kwargs2)


def compute_sigma(velocity_file, ofname, tdir):
    """Computes sigma for ItsLIVE files
    velocity_file:
        Input file (result of merge_to_pism_rule()) with multiple timesteps
    ofname:
        Output file
    """
    # Create output file, based on input file
    with netCDF4.Dataset(velocity_file) as ncin:
        schema = ncutil.Schema(ncin)
        ns481_grid = ncin.grid

        for vname in ('u_ssa_bc', 'v_ssa_bc', 'v'):
            if vname in schema.vars:
                del schema.vars[vname]

        ntime = len(ncin.dimensions['time'])
        #nc481_grid = ncin.ns481_grid


    # Initialize output file
    with netCDF4.Dataset(ofname, 'w') as ncout:
        # Create output file, copying struture of input
        schema.create(ncout)
        var_kwargs = {'zlib': True}

        # Add new variables
        for ix in (0,1):
            ncv = ncout.createVariable(f'strain_rates_{ix}', 'd', ('time', 'y', 'x'), **var_kwargs)
            ncv.long_name = f'Eigenvalue #{ix} of strain rate, used to compute von Mises stress; see glacier.von_mises_stress_eig()'
        ncv = ncout.createVariable('mask', 'i', ('time', 'y', 'x'), **var_kwargs)
        ncv.long_name = 'PISM Mask'
        ncv.description = '0=bare ground; 2=grounded ice; 3=floating ice; 4=open water'
        ncv = ncout.createVariable(f'sigma', 'd', ('time', 'y', 'x'), **var_kwargs)
        ncv.long_name = 'von Mises Stress'
        ncv.description = 'Computed using glacier.von_mises_stress_eig()'
        ncv.units = 'Pa'



    # Generate sigmas for each timestep
    for itime in range(ntime):
        #of_tmp = tdir.filename()
        of_tmp = './tmp.nc'
        get_von_Mises_stress(
            ns481_grid, velocity_file,
            of_tmp, tdir,
            itime=itime,
            dry_run=False,
            sigma_max=1.e6,    # Doesn't matter
            dt_s=100.)            # Really short

        # Copy from temporary file into final output file
        eig = list()
        with netCDF4.Dataset(ofname, 'a') as ncout:
            with netCDF4.Dataset(of_tmp) as ncin:
                for ix in (0,1):
                    L = ncin.variables['strain_rates[{}]'.format(ix)][:]
                    eig.append(L)
                    ncv = ncout.variables['strain_rates_{}'.format(ix)]
                    ncv[itime,:] = L

                ncout.variables['sigma'][itime,:] = glacier.von_mises_stress_eig(*eig)

                ncout.variables['mask'][itime,:] = ncin.variables['mask'][0,:]

def compute_sigma_rule(itslive_nc, odir):

    """Runs the PISM compute-sigma calculation on a merged localized
    ItsLIVE file.

    itslive_nc:
        Result of itslive.merge_to_pism_rule()
    odir:
        Directory to place results

    """
    ofname = make.opath(itslive_nc, odir, '_sigma')
    def action(tdir):
        compute_sigma(itslive_nc, ofname, tdir)
    return make.Rule(action,
        [itslive_nc], [ofname])


