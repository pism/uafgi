import collections
import traceback
import os,sys
import numpy as np
import pandas as pd
import datetime
from uafgi import argutil,gdalutil,glacier,bedmachine,ncutil,make,cfutil
import uafgi.data
from uafgi.pism import pismutil
from uafgi.pism import calving0
import netCDF4
import PISM
import shapely.geometry
import traceback

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



def run_pism(bedmachine_file0, ns481_grid, fjord_classes, velocity_file, output_file, tdir, dry_run=False, attrs=dict(),
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

#    bedmachine_file0 = uafgi.data.bedmachine_local(grid)


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
    bedmachine_file0 = uafgi.data.bedmachine_local(ns481_grid)
    args = (bedmachine_file0, ns481_grid, None, velocity_file, output_file, tdir)
    kwargs2 = kwargs.copy()
    kwargs2['remove_downstream_ice'] = False    # Don't cut off any ice below the terminus
    kwargs2['dt_s'] = 100.                     # Run only briefly
    kwargs2['delete_vars'] = {'ice_area_specific_volume', 'thk', 'total_retreat_rate', 'flux_divergence'}

    return run_pism(*args, **kwargs2)

# ==========================================================================
def get_von_Mises_stress_gimpdem(ns481_grid, velocity_file, output_file, tdir,
    dry_run=False, attrs=dict(),
    itime=None, year=None,
    **pism_kwargs0):

    """Clean sheet replacement for get_von_Mises_stress().
    Also incorporates use of the GIMP DEM in place of Bedmachine's elevation.

    Does a single PISM run
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



    bedmachine_file0 = uafgi.data.bedmachine_local(ns481_grid)
    gimpdem_file0 = uafgi.data.gimpdem_local(ns481_grid)

    # Run only briefly
    dt_s = 100.
    # PISM Variables not needed in final output
    delete_vars = {'ice_area_specific_volume', 'thk', 'total_retreat_rate', 'flux_divergence'}


    # ============ Determine input/output filenames

    # Determine the local grid
    grid_file = uafgi.data.measures_grid_file(ns481_grid)
    grid_info = gdalutil.FileInfo(grid_file)


    # Use to construct Makefile rules
    if dry_run:
        inputs = [grid_file, bedmachine_file0, gimpdem_file0, velocity_file]
        outputs = [output_file_raw]
        return inputs, outputs

    # ============================================


    # Get total kwargs to use for PISM
    default_kwargs = dict(calving0.FrontEvolution.default_kwargs.items())
    default_kwargs['min_ice_thickness'] = 50.0    # See TODO below
    kwargs = argutil.select_kwargs(pism_kwargs0, default_kwargs)


    # -------------------------------------------------
    # Construct an appropriate Bedmachine file
    bedmachine_file1 = tdir.filename(suffix='.nc')

    # Use Gimp DEM Elevations
    with netCDF4.Dataset(gimpdem_file0, 'r') as nc:
        elevation = nc.variables['elevation'][:]
    # ...merged into Bedmachine file
    with netCDF4.Dataset(bedmachine_file0, 'r') as ncin:
        schema = ncutil.Schema(ncin)
        with netCDF4.Dataset(bedmachine_file1, 'w') as ncout:
            # Create all variables
            schema.create(ncout)

            # Copy all variables except thickness
            del schema.vars['thickness']
            schema.copy(ncin, ncout)

            # Compute our own thickness
            ncout.variables['thickness'][:] = elevation - ncin.variables['bed'][:]
    # -------------------------------------------------
    # Obtain start and end time in PISM units (seconds)
    with netCDF4.Dataset(velocity_file) as nc:
        nctime = nc.variables['time']
        time_units = cf_units.Unit(nctime.units, nctime.calendar)
        time_units_s = cfutil.replace_reftime_unit(time_units, 'seconds')
        t0_py = time_units.num2date(nctime[0])    # Python-format date
        t0_s = time_units_s.date2num(t0_py)
        t1_s = t0_s + dt_s
    # ---------------------------------------------------------------

    print('============ Running time {} for {} seconds'.format(t0_py, t1_s-t0_s))
    output_file3 = tdir.filename()
    print('     ---> {}'.format(output_file3))
    sys.stdout.flush()

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

        # Add parameter info
        nc.creator = 'flow_simulation.py'
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



# ==============================================================================

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
        of_tmp = tdir.filename()
        #of_tmp = './tmp.nc'
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

# ===============================================================
FlowRateRet = collections.namedtuple('FlowRateRet', ('flux', 'up_area'))
def flow_rate(grid, fjord_gd, up_loc_gd, terminus, uu, vv, mask, itslive_nc=None, debug_out_nc=None):
    """
    grid:
        Name of local grid to use (eg: W69.10N)
    fjord_gd: np.array bool
        True where there is fjord (or glacier grounded below sea level)
        NOTE: This is as read by GDAL (_gd); may be flipped up/down from other arrays.
    up_loc_gd: Shapely Point
        A point on the glacier, upstream of the terminus.
        NOTE: This is as read by GDAL (_gd), to fit into fjord array.
    terminus: Shapely Line
        The glacier terminus (used to cut glacier)
    itslive_nc_tpl:
        Template to give the Velocities file to use.
    uu, vv: np.arry [m s-1]
        Flow we're integrating
    mask: np.array
        PISM mask
    itslive_nc: str (OPTIONAL)
        Name of input velocities file
        If set, used only for debugging
    """


    # Determine the fjord
    grid_info = gdalutil.FileInfo(uafgi.data.measures_grid_file(grid))

    # Cut off the fjord at our terminus
    fjc_gd = glacier.classify_fjord(fjord_gd, grid_info, up_loc_gd, terminus)
    fjc = np.flipud(fjc_gd)    # fjord was originally read by gdal, it is flipped u/d
    up_area = np.sum(np.isin(fjc, glacier.GE_TERMINUS)) * grid_info.dx * grid_info.dy    # [m^2]
    fjord = np.isin(fjc, glacier.ALL_FJORD)
#    print('up_area: {}: {}'.format(up_area, terminus))
#    print('up_area: {}'.format(up_area))

    # Update the PISM map to reflect cut-off fjord.
    #ICE_TYPES = (pismutil.MASK_GROUNDED, pismutil.MASK_FLOATING)
    kill_mask = np.logical_and(np.isin(mask, pismutil.MULTIMASK_ICE), fjc==glacier.LOWER_FJORD)
    mask[kill_mask] = pismutil.MASK_ICE_FREE_OCEAN

    kill_mask = (mask == pismutil.MASK_ICE_FREE_OCEAN)
    uu[kill_mask] = 0
    vv[kill_mask] = 0

    # Compute flux across the boundary (and length of the boundary)
    flux = pismutil.flux_across_terminus(mask, fjord, uu, vv, grid_info.dx, grid_info.dy)

    # ------------------------------------------------------------------------
    if itslive_nc is not None:
        with netCDF4.Dataset(itslive_nc) as ncin:
            schema = ncutil.Schema(ncin)
            schema.keep_only_vars('x', 'y')

            with netCDF4.Dataset(debug_out_nc, 'w') as ncout:
                schema.create(ncout, var_kwargs={'zlib': True})
                for vname in ('fjc', 'fjord', 'mask', 'uu', 'vv', 'flux'):
                    ncout.createVariable(vname, 'd', ('y','x'))
                schema.copy(ncin, ncout)
                ncout.variables['fjc'][:] = fjc[:]
                ncout.variables['fjord'][:] = fjord[:]
                ncout.variables['mask'][:] = mask[:]
                ncout.variables['uu'][:] = uu[:]
                ncout.variables['vv'][:] = vv[:]
                ncout.variables['flux'][:] = flux[:]
    # ------------------------------------------------------------------------

    tflux = np.sum(flux)    # Flux across boundary [m^2 s-1]
    
    return FlowRateRet(tflux, up_area)

def flow_rate2(row, w21t, Merger, year0, year1):
    """Simplified API for multiple termini: computes velocity and
    velocity*sigma flux for a set of termini over time.

    row:
        Row from the "select" dataframe of our glaciers of interest.
    w21t:
        Available termini for all glaciers.
         = uafgi.data.w21.read_termini(uafgi.data.wkt.nsidc_ps_north).df
    Merger: itslive.[ItsliveMerger/W21Merger]
        Which type of velocity file we'll use
    year0, year1:
        Year range.  Must match names of velocity files.
    """
    orows = list()

    grid = row['ns481_grid']
    fjord_gd = np.isin(row['fjord_classes'], glacier.ALL_FJORD)    # _gd == "as read by GDAL"
    up_loc_gd = row['up_loc']

    # Get termini for that
    w21tx = w21t[w21t['w21t_Glacier'] == row['w21t_Glacier']].sort_values(['w21t_date'])


    itslive_nc = Merger.ofname(grid, year0, year1)
    sigma_nc = os.path.splitext(itslive_nc)[0] + '_sigma.nc'

    with netCDF4.Dataset(itslive_nc) as nc:
        time = cfutil.read_time(nc, 'time')
        nct = nc.variables['time']
        time_bnds = cfutil.read_time(nc, 'time_bnds',
            units=nct.units, calendar=nct.calendar)

        # Fix end of range in Its-Live
        time_bnds[:,1] += Merger.time_bnds_adjust

    for target_year in range(year0, year1):
        try:
            target_dt = datetime.datetime(target_year, 9, 1)

            # Figure out time index in velocity file
            for itime_itslive in range(time_bnds.shape[0]):
                if target_dt >= time_bnds[itime_itslive,0] and target_dt < time_bnds[itime_itslive,1]:
                    break

            # Read velocity components
            with netCDF4.Dataset(itslive_nc) as nc:
                uu = ncutil.convert_var(nc, 'u_ssa_bc', 'm s-1')[itime_itslive,:]
                vv = ncutil.convert_var(nc, 'v_ssa_bc', 'm s-1')[itime_itslive,:]

            # Read sigma
            with netCDF4.Dataset(sigma_nc) as nc:
                sigma = nc.variables['sigma'][itime_itslive,:]
                mask = nc.variables['mask'][itime_itslive,:]

                fjord_width = row['w21_mean_fjord_width'] * 1000.    # Data in km


            # Figure out terminus trace closest to target date
            df = w21tx.copy()
            if len(df) == 0:
                continue

            df['dtdiff'] = df['w21t_date'].apply(lambda dt: abs((dt - target_dt).total_seconds()))
            terminus_row = df.loc[df['dtdiff'].idxmin()]
            print('target: {} -> ({} {})'.format(target_dt, terminus_row['w21t_Year'], terminus_row['w21t_Day_of_Yea']))
            terminus = terminus_row.w21t_terminus

            # Compute velocity flux
            fr_aflux = flow_rate(grid, fjord_gd, up_loc_gd, terminus,
                uu, vv,
                mask, fjord_width, itslive_nc=itslive_nc, debug_out_nc='debug_aflux.nc')

            # Compute sigma flux
            fr_sflux = flow_rate(grid, fjord_gd, up_loc_gd, terminus,
                uu*sigma, vv*sigma,
                mask, fjord_width, itslive_nc=itslive_nc, debug_out_nc='debug_sflux.nc')

            orow = {'w21t_key': row['w21t_key'], 'year': target_year, 'velocity_source': Merger.__name__, 'up_area':fr_aflux['up_area'], 'aflux': fr_aflux['flux'], 'sflux': fr_sflux['flux']}
            orows.append(orow)
        except:
            traceback.print_exc()
            pass

    return pd.DataFrame(data=orows)


FlowRate3Ret = collections.namedtuple('FlowRateRet3', ('aflux', 'sflux', 'up_area'))
def flow_rate3(grid, bedmachine_file, fj_poly, velocity_file, sigma_file, itime, termini, up_loc_gd):
    """
    velocity_file:
        Raster of ice surface velocities
    sigma_file:
        Raster of PISM-computed sigma values
        (Derived from sigma_file)
    itime:
        Time index in velocity/sigma files to use
    terminus: shapely.LineString, ...
        The termini across which to integrate fluxes
#    fjord_gd:
#        Fjord raster
#        Eg: np.isin(selrow['fjord_classes'], glacier.ALL_FJORD)
#        NOTE: _gd == "as read by GDAL"
    up_loc_gd:
        Location of upstream point in fjord
        eg: selrow['up_loc']
        NOTE: _gd == "as read by GDAL"
    Returns: [(aflux, sflux, up_area), ...]
        
    """
    sigma_file = os.path.splitext(velocity_file)[0] + '_sigma.nc'

    # Load the fjord
    fjord_gd = bedmachine.get_fjord_gd(bedmachine_file, fj_poly)


    # Read velocity components
    with netCDF4.Dataset(velocity_file) as nc:
        uu = ncutil.convert_var(nc, 'u_ssa_bc', 'm s-1')[itime,:]
        vv = ncutil.convert_var(nc, 'v_ssa_bc', 'm s-1')[itime,:]

    # Read sigma
    with netCDF4.Dataset(sigma_file) as nc:
        sigma = nc.variables['sigma'][itime,:]
        mask = nc.variables['mask'][itime,:]

    ret = list()
    for terminus in termini:
        # Turn debugging on/off
        debug_kwargs = dict(itslive_nc=velocity_file, debug_out_nc='debug_sflux.nc')
        #debug_kwargs = dict()

        # Compute velocity flux
        fra = flow_rate(grid, fjord_gd, up_loc_gd, terminus,
            uu, vv,
            mask, **debug_kwargs)

        # Compute sigma flux
        frs = flow_rate(grid, fjord_gd, up_loc_gd, terminus,
            uu*sigma, vv*sigma,
            mask, **debug_kwargs)

        answer = FlowRate3Ret(fra.flux, frs.flux, fra.up_area)
#        print(answer.sflux/answer.aflux, answer)
        ret.append(answer)

    return ret

