import sys
import numpy as np
import os
import re
import pandas as pd
import pyproj
from uafgi import gdalutil,ogrutil,shputil
from uafgi import pdutil,shapelyutil
import shapely
import shapely.geometry
from osgeo import ogr,osr
import uafgi.data
import uafgi.data.bkm15
import uafgi.data.cf20
import uafgi.data.fj
import uafgi.data.m17
import uafgi.data.mwp
import uafgi.data.ns481
import uafgi.data.ns642
import uafgi.data.w21 as d_w21
from uafgi.data import d_sl19
import uafgi.data.wkt
from uafgi.data import greenland,stability
import pickle
from uafgi import bedmachine,glacier,cartopyutil,cptutil,dtutil
import collections
import scipy.stats
import netCDF4

def select_glaciers():
    """Step 1: Determine a set of glaciers for our experiment.

    Returns: A dataframe with columns:
    Index(['w21t_Glacier', 'w21t_date_termini', 'w21t_glacier_number', 'w21t_tloc',
           'w21_popular_name', 'w21_greenlandic_name', 'w21_coast', 'w21_category',
           'w21_Qr', 'w21_Qf', 'w21_Qm', 'w21_Qs', 'w21_Qc_inferred', 'w21_qm',
           'w21_qf', 'w21_qc', 'w21_mean_depth', 'w21_min_depth',
           'w21_quality_str', 'w21_area_grounded_1992_2017',
           'w21_area_grounded_1992_1997', 'w21_area_grounded_1998_2007',
           'w21_area_grounded_2008_2017', 'w21_mean_fjord_width',
           'w21_length_grounded_1992_2017', 'w21_length_grounded_1992_1997',
           'w21_length_grounded_1998_2007', 'w21_length_grounded_2008_2017',
           'w21_ocean_model_sample_area', 'w21_mean_TF_1992-2017',
           'w21_mean_TF_1992-1997', 'w21_mean_TF_1998-2007',
           'w21_mean_TF_2008-2017', 'w21_subglacial_discharge_1992_2017',
           'w21_subglacial_discharge_1992_1997',
           'w21_subglacial_discharge_1998_2007',
           'w21_subglacial_discharge_2008_2017', 'w21_mean_xsection_area',
           'w21_mean_undercutting_1992_2017', 'w21_mean_undercutting_1992_1997',
           'w21_mean_undercutting_1998_2007', 'w21_mean_undercutting_2008_2017',
           'w21_mean_undercutting_uncertainty', 'w21_flux_basin_mouginot_2019',
           'w21_mean_discharge', 'w21_mean_mass_balance',
           'w21_reference_smb_1961_1990', 'w21_glacier_number', 'w21_data_fname',
           'w21_key', 'w21_allnames', 'w21_tloc', 'fj_poly', 'fj_fid', 'ns481_key',
           'ns481_grid', 'ns481_poly', 'up_key', 'up_fid', 'up_id', 'up_loc',
           'cf20_key', 'cf20_glacier_id', 'cf20_greenlandic_name',
           'cf20_official_name', 'cf20_alt_name', 'cf20_ref_name', 'cf20_fname',
           'cf20_uniqename', 'cf20_locs', 'cf20_allnames', 'ns642_key',
           'ns642_GlacierID', 'ns642_date_termini', 'ns642_points'],
          dtype='object')
    """

    map_wkt = uafgi.data.wkt.nsidc_ps_north

    pd.set_option('display.max_columns', None)

    # Read user overrides of joins and columns
    over = stability.read_overrides()

    # Get a single terminus point for each glacier; our initial selection
    w21t = d_w21.read_termini(map_wkt)

    # ------------------- Remove blackout glaciers
    # Set up glacers we DON'T want to select
    w21t_blackouts = pd.DataFrame({
        'w21t_Glacier' : [
            # Kakivfaat glacier has one glacier front in Wood data, but two in NSIDC-0642 (as it separates)
            'Kakivfaat',
            # Glacier has small fjord, terminus is bigger than the fjord, has already retreated back to land
            'Helland W',
        ]
    })

    # Get the original w21.df's index onto the blackouts list
    blackouts_index= pd.merge(w21t.df.reset_index(), w21t_blackouts, how='inner', on='w21t_Glacier').set_index('index').index

    # Remove items we previously decided we DID NOT want to select
    w21t.df = w21t.df.drop(blackouts_index, axis=0)
    # ---------------------------------
    w21t.df = w21t.df[w21t.df.w21t_Glacier != 'Kakivfaat']
#    w21tx_termini = pdutil.group_and_tuplelist(w21t.df, 'w21t_Glacier', [ ('terminus_by_date', ['w21t_date', 'w21t_terminus']) ])
    w21tx = d_w21.termini_by_glacier(w21t)

    # Convert [(date, terminus), ...] into a MultiPoint object
    w21tx.df['w21t_points'] = w21tx.df['w21t_date_termini'].map(
            lambda date_terminus_list: shapelyutil.pointify(shape for _,shape in date_terminus_list))
    # Get the centroid of the multipoint; now it's some sort of point within the temrinus region
    w21tx.df['w21t_tloc'] = w21tx.df['w21t_points'].map(lambda mpt: mpt.centroid)

    # Join back with w21
    w21 = d_w21.read(map_wkt)
    w21tx.df = pdutil.merge_nodups(w21tx.df, w21.df, how='left', left_on='w21t_glacier_number', right_on='w21_glacier_number')
    # Now we have column w21_key

    # Identify the hand-drawn fjord matched to each glacier in our selection
    fj = uafgi.data.fj.read(uafgi.data.wkt.nsidc_ps_north)
    ret = uafgi.data.fj.match_to_points(
        w21tx, 'w21t_glacier_number', 'w21t_points',
        fj, debug_shapefile='fjdup.shp')
    select = ret['select']
    select.df = select.df.drop('w21t_points', axis=1)    # No longer need, it takes up space

    # PS: if there's a problem, ret['glaciersdup'] and ret['fjdup'] will be set; see 'fjdup.shp' file.
    select.df = select.df.dropna(subset=['fj_fid'])    # Drop glaciers without a hand-drawn fjord

    # ----------------------------------------------------------
    # Obtain set of local MEAUSRES grids on Greenland
    ns481 = uafgi.data.ns481.read(uafgi.data.wkt.nsidc_ps_north)

    match = pdutil.match_point_poly(select, 'w21t_tloc', ns481, 'ns481_poly',
        left_cols=['fj_poly'], right_cols=['ns481_poly'])
    match.df['fjord_grid_overlap'] = match.df.apply(
            lambda x: 0 if (type(x['ns481_poly'])==float or type(x['fj_poly']) == float)
            else x['ns481_poly'].intersection(x['fj_poly']).area / x['fj_poly'].area,
            axis=1)

    try:
        select = match.left_join(overrides=over[['w21_key', 'ns481_key']])
    except pdutil.JoinError as err:
        print(err.df[['w21_key', 'ns481_key']].sort_values('ns481_key'))
        raise

    # Only keep glaciers inside a MEASURES grid
    select.df = select.df[~select.df['ns481_key'].isna()]

    # ----- Add a single upstream point for each glacier
    up = shputil.read_df(
        uafgi.data.join('upstream/upstream_points.shp'),
        wkt=uafgi.data.wkt.nsidc_ps_north, shape='loc', add_prefix='up_')

    match = pdutil.match_point_poly(up, 'up_loc', select, 'fj_poly').swap()

    try:
        select = match.left_join()
    except pdutil.JoinError as err:
        df = err.df[['w21t_key']].drop_duplicates()
        df = pdutil.merge_nodups(df, select.df[['w21t_key','fj_poly', 'fj_fid']], how='left', on='w21t_key')
        # Write fjords with duplicate upstream points to a shapefile
        fields=[ogr.FieldDefn('fj_fid',ogr.OFTInteger)]
        shputil.write_shapefile2(df['fj_poly'].tolist(), 'fjupdup.shp', fields, attrss=list(zip(df['fj_fid'].tolist())))

        # Print it out:
        print('Fjords with duplicate upstream points:\n{}'.format(df))

    # -------------------------------------------------------------
    # Join with bkm15
    bkm15 = uafgi.data.bkm15.read(uafgi.data.wkt.nsidc_ps_north)
    match = pdutil.match_point_poly(bkm15, 'bkm15_loc', select, 'fj_poly', left_cols=['bkm15_allnames']).swap()
    select = match.left_join(overrides=over)

# Not sure what purpose this serves.
#    select.df = pdutil.override_cols(select.df, over, 'w21_key',
#        [('bkm15_lat', 'lat', 'lat'),
#         ('bkm15_lon', 'lon', 'lon'),
#         ('bkm15_loc', 'loc', 'loc')])

    # ----------------------------------------------------------


    # Get historical termini


    # ---- Join with CALFIN dataset high-frequency termini
    cf20= uafgi.data.cf20.read(uafgi.data.wkt.nsidc_ps_north)
    print('===================================')
    match = pdutil.match_point_poly(cf20, 'cf20_locs', select, 'fj_poly').swap()
    try:
        select = match.left_join(overrides=over)
    except pdutil.JoinError as err:
        print(err.df)
        raise

    # ----- Join with NSIDC-0642 (MEASURES) annual termini
    ns642 = uafgi.data.ns642.read(uafgi.data.wkt.nsidc_ps_north)
    # Combine all points for each GlacierID

    ns642x = uafgi.data.ns642.termini_by_glacier(ns642)

    # Convert [(date, terminus), ...] into a MultiPoint object
    ns642x.df['ns642_points'] = ns642x.df['ns642_date_termini'].map(
            lambda date_terminus_list: shapelyutil.pointify(shape for _,shape in date_terminus_list))

    print('******** ns642x {}'.format(ns642x.df.columns))

    match = pdutil.match_point_poly(
        ns642x, 'ns642_points', select, 'fj_poly').swap()
    select = match.left_join(overrides=over)

    # ------ Join with Slater et al (2019) work
    sl19 = d_sl19.read(map_wkt)

    # Join on bkm15_id to get sl19_rignotid
#    print('AA1 ', len(select.df))
    select.df = pd.merge(select.df, sl19.df[['sl19_bjorkid','sl19_rignotid']].dropna(), how='left', left_on='bkm15_id', right_on='sl19_bjorkid')

#    print('AA2 ', len(select.df))
    # Add in sl19_rignotid from overrides
    select.df = pd.merge(select.df, over[['w21_key', 'sl19_rignotid']].dropna(), how='left', on='w21_key', suffixes=(None,'_r'))
    col = select.df['sl19_rignotid']    # this col has precedence
    select.df['sl19_rignotid'] = col.combine_first(select.df['sl19_rignotid_r'])
    select.df = select.df.drop(['sl19_rignotid_r'], axis=1)    

#    select = pdutil.merge_nodups(select, over[['w21_key', 'sl19_rignotid']], on='w21_key')
#    print('AA3 ', len(select.df))


    _df = sl19.df[['sl19_rignotid', 'sl19_lon', 'sl19_lat', 'sl19_loc',
        'sl19_termpos_L', 'sl19_termpos_t',

        # Subglacial discharge Q from RACMO (p. 2492)
        'sl19_RACMO_t', 'sl19_RACMO_Q', 'sl19_RACMO_tJJA', 'sl19_RACMO_QJJA', 'sl19_RACMO_Qbaseline',


        'sl19_EN4_TF',
        'sl19_EN4_t', 'sl19_EN4_TFbaseline', 'sl19_sector', 'sl19_melt_t',
        'sl19_melt_m', 'sl19_melt_meltbaseline', 'sl19_iceflux_enderlin',
        'sl19_iceflux_king', 'sl19_iceflux_final', 'sl19_key']]

    select.df = pdutil.merge_nodups(select.df, _df, how='left', on='sl19_rignotid')
#    print('AA4 ', len(select.df))

    return select

def retreat_history(select):
    """Generates a dataframe of the retreat history (in terms of fjord area) of each glacier.
    select: ExtDf
        Output of select_glaciers() above
    """
#    for _,row in select.df.iterrows():
#        if pd.isna(row.up_loc):
#        print(row.w21t_Glacier, row.w21t_glacier_number, row.ns481_grid, row.up_loc)

#    on = False
    data = list()
    for _,row in select.df.iterrows():
        if row.w21t_Glacier in {'Helland W'}:    # Degenerate
            continue
#
#        if row.w21t_glacier_number == 175:
#            on = True
#        if not on:
#            continue

        print('********* {} {} {}'.format(row.ns481_grid, row.w21t_glacier_number, row.w21t_Glacier))
        # Retrieve columns we need
        dtterm = sorted(row['w21t_date_termini'])

        # Determine the local grid
        grid = row['ns481_grid']
        grid_file = uafgi.data.measures_grid_file(grid)
        grid_info = gdalutil.FileInfo(grid_file)
        bedmachine_file = uafgi.data.bedmachine_local(grid)

        # Load the fjord
        fjord = bedmachine.get_fjord(bedmachine_file, row.fj_poly)

        # Quantify retreat for each terminus
        for dt,terminus in dtterm:
            fjc = glacier.classify_fjord(fjord, grid_info, row.up_loc, terminus)
            up_fjord = np.isin(fjc, glacier.GE_TERMINUS)
            up_area = np.sum(np.sum(up_fjord)) * grid_info.dx * grid_info.dy
#            print('   up_area: ', np.sum(np.sum(up_fjord)), grid_info.dx, grid_info.dy)
            data.append((row['w21t_glacier_number'], dt, up_area))

    df = pd.DataFrame(data, columns=['w21t_glacier_number', 'date', 'up_area'])
    df.to_pickle('retreat_area.df')
    return df
#            print(row['w21t_Glacier'], row['w21t_glacier_number'], dt, up_area)



def compute_sigma(velocity_file, bm_file, ofname, tdir):


#***** We want to:
#
#1. Do not worry about cutting off at the terminus.  We want to compute sigma for ALL avialable areas with ice and velocity measurements.
#2. THEN we compute sigma, integrated over the terminus.


    """Computes sigma for ItsLIVE files
    velocity_file:
        Input file (result of merge_to_pism_rule()) with multiple timesteps
    ofname:
        Output file
    """


    # --------------------------------------------------------
    # Create output file, based on input velocity file

    # Copy input file to output; but not the velocity variables
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
    # --------------------------------------------------------



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


# --------------------------------------------------------------------------
# Helper functions for FitSlaterResidual.compute()

def _binme(t,X,bins):
    return scipy.stats.binned_statistic(t,X, statistic='mean', bins=bins).statistic

def _bin_multi(bins, t_datas, detrend=False):
    """t_datas: [(X_t, X_data), (Y_t, Y_data), ...]
    """
    bbins = .5*(bins[:-1]+ bins[1:])
    mask_in = np.ones(len(bbins), dtype=bool)
    X_bs = list()
    for X_t, X_data in t_datas:
        X_b = _binme(X_t, X_data, bins)
        X_bs.append(X_b)
        mask_in = mask_in & ~np.isnan(X_b)

    X_bs = [X_b[mask_in] for X_b in X_bs]
    bbins = bbins[mask_in]
    if detrend:
        X_bs = [scipy.signal.detrend(X_b) for X_b in X_bs]

    return X_bs,bbins


def get_glacier_df(w21t_glacier_number, w21_mean_fjord_width, velterm_df):
    """Gets the portion of velterm_df for one glacier.
    velterm_df:
        Result of d_velterm.read()"""

    #### Wood et al + CALFIN dataset
    df = velterm_df[velterm_df.glacier_id == w21t_glacier_number]

    # ---------------------------------------------------
    # Get rid of future termini
    df = df[df.term_year < 2020]

    # Useonly termini since 2000
    df = df[(df['term_year'] > 2000) & (df.term_year < 2020)]

    # Use only velocities older than the terminus
    df = df[df.vel_year < df.term_year]

    # Convert up_area to up_len_km
    df['up_len_km'] = df['up_area'] / (w21_mean_fjord_width * 1e6)

    return df

def slater_termpos_regression(selrow, glacier_df, year_range):

    bins = np.arange(*year_range, 1)    # binsize = 1 year

    # Extract dataframe showing just year vs. terminus location
    yearlen = glacier_df.sort_values(['term_year'])[['term_year', 'up_len_km']].drop_duplicates()

    # Bin by 1-year intervals
    (termpos_b1,up_len_km_b1),bbins1 = _bin_multi(bins, [
        (selrow.sl19_termpos_t, selrow.sl19_termpos_L),
        (yearlen.term_year, yearlen.up_len_km)])

    # Get correspondence between Slater and our terminus positions
    # Allows prediction of termpos from up_len_km.
    # The regression is done based on 1-year bins (i.e. no binning)
    termpos_lr = scipy.stats.linregress(up_len_km_b1,termpos_b1)

    return up_len_km_b1,termpos_b1,bbins1,termpos_lr


# ---------------------------------------------------
FitSlaterResidualsRet = collections.namedtuple('FitSlaterResidualsRet', (
    # 1-year binned data
    "bbins1", 'termpos_b1', 'up_len_km_b1',

    "bbins1l", 'melt_b1l', 'termpos_b1l',

    # 5-year binned data
    'bbins', 'melt_b', 'termpos_b',

    # Linear regressions
    'termpos_lr',     # Conversion from up_len_km to Slater's termpos
    'slater_lr',        # melt vs. termpos for this glacier, as per Slater et al 2019
    'resid_lr',        # Our sigma ("fluxratio") vs. residual of Slater's prediction

    # Data points per terminal/velocity pair
     'glacier_df',

    #...and grouped by year
    'resid_df'))

def fit_slater_residuals(selrow, velterm_df):
    """Computes fit between the sigma values from velterm, and residuals
    of Slater's predictions vs. reality.

    selrow:
        A row from dataframe returned by `select_glacers()`
        (above); usually stored on disk and retrieved.
    velterm_df:
        Result of d_velterm.read()
    Returns: FitSlaterResidualRet
    """

    year_range = (1960,2021)
    bins = np.arange(*year_range,5)    # binsize=5

    glacier_df = get_glacier_df(selrow.w21t_glacier_number, selrow.w21_mean_fjord_width, velterm_df)

    # Reproduce plots in Slater et al 2019 supplement

    # ----------- Bin by 5-year intervals
    (melt_b,termpos_b),bbins = _bin_multi(bins, [
        (selrow.sl19_melt_t, selrow.sl19_melt_m),
        (selrow.sl19_termpos_t, selrow.sl19_termpos_L)])

    # -------- Get conversion from our up_len_km to Slater's termpos
    bins1 = np.arange(*year_range, 1)    # binsize = 1 year

    # Extract dataframe showing just year vs. terminus location
    yearlen = glacier_df.sort_values(['term_year'])[['term_year', 'up_len_km']].drop_duplicates()

    # Get correspondence between Slater and our terminus positions
    # Allows prediction of termpos from up_len_km.
    # The regression is done based on 1-year bins (i.e. no binning)
    # Bin by 1-year intervals
    (termpos_b1,up_len_km_b1),bbins1 = _bin_multi(bins1, [
        (selrow.sl19_termpos_t, selrow.sl19_termpos_L),
        (yearlen.term_year, yearlen.up_len_km)])
    termpos_lr = scipy.stats.linregress(up_len_km_b1,termpos_b1)


    # Bin 1-year for plotting
    (melt_b1l,termpos_b1l),bbins1l = _bin_multi(bins1, [
        (selrow.sl19_melt_t, selrow.sl19_melt_m),
        (selrow.sl19_termpos_t, selrow.sl19_termpos_L)])

    # ----------------

    # Determine Slater's melt value at each of our data's time points
    melt_sp = scipy.interpolate.UnivariateSpline(bbins,melt_b)
    glacier_df['sl19_melt'] = glacier_df['term_year'].map(melt_sp)

    # Predicted termpos based on Slater's relationship with melt
    slater_lr = scipy.stats.linregress(melt_b,termpos_b)
    glacier_df['sl19_pred_termpos'] = glacier_df['sl19_melt'] * slater_lr.slope + slater_lr.intercept

    # Conversion of our up_len_km to Slater units (to use in melt vs. terminal position relation)
    glacier_df['our_termpos'] = glacier_df['up_len_km'].map(lambda x: termpos_lr.slope*x + termpos_lr.intercept)

    # Difference between our termpos and the predicted termpos
    # This is the residual between Slater's prediction vs. what actually happened
    glacier_df['termpos_residual'] = glacier_df['our_termpos'] - glacier_df['sl19_pred_termpos']


    # -----------------------------------------------
    # See if there's a correlation between residuals on terminal position,
    # and our computed sigma (based on fjord geometry)
    resid_df = glacier_df[['term_year', 'fluxratio', 'termpos_residual']].dropna().groupby('term_year').mean().reset_index()
    resid_lr = scipy.stats.linregress(resid_df.fluxratio, resid_df.termpos_residual)
    #print(resid_lr)

    # -----------------------------------------------
    # Store outputs
    return FitSlaterResidualsRet(
        bbins1, termpos_b1, up_len_km_b1,
        bbins1l, melt_b1l, termpos_b1l,
        bbins, melt_b, termpos_b,
        termpos_lr, slater_lr, resid_lr,
        glacier_df, resid_df)

# ----------------------------------------------------------------
def plot_reference_map(fig, selrow):
    """Plots a reference map of a single glacier

    fig:
        Pre-created figure (of a certain size/shape) to populate.
    selrow:
        Row of d_stability.read()"""

    # fig = matplotlib.pyplot.figure()

    # -----------------------------------------------------------
    # (1,0): Map
    # Get local geometry
    bedmachine_file = uafgi.data.join_outputs('bedmachine', 'BedMachineGreenland-2017-09-20_{}.nc'.format(selrow.ns481_grid))
    with netCDF4.Dataset(bedmachine_file) as nc:
        nc.set_auto_mask(False)
        mapinfo = cartopyutil.nc_mapinfo(nc, 'polar_stereographic')
        bed = nc.variables['bed'][:]
        xx = nc.variables['x'][:]
        yy = nc.variables['y'][:]

    # Set up the basemap
    ax = fig.add_axes((.1,.1,.9,.9), projection=mapinfo.crs)
    #ax = fig.add_subplot(spec[2,:], projection=mapinfo.crs)
    ax.set_extent(mapinfo.extents, crs=mapinfo.crs)
    ax.coastlines(resolution='50m')


    # Plot depth in the fjord
    fjord_gd = bedmachine.get_fjord_gd(bedmachine_file, selrow.fj_poly)
    fjord = np.flip(fjord_gd, axis=0)
    bedm = np.ma.masked_where(np.logical_not(fjord), bed)

    bui_range = (0.,350.)
    cmap,_,_ = cptutil.read_cpt('caribbean.cpt')
    pcm = ax.pcolormesh(
        xx, yy, bedm, transform=mapinfo.crs,
        cmap=cmap, vmin=-1000, vmax=0)
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label('Fjord Bathymetry (m)')
        
    # Plot the termini
    date_termini = sorted(selrow.w21t_date_termini)

    yy = [dtutil.year_fraction(dt) for dt,_ in date_termini]
    year_termini = [(y,t) for y,(_,t) in zip(yy, date_termini) if y > 2000]

    for year,term in year_termini:
        ax.add_geometries([term], crs=mapinfo.crs, edgecolor='red', facecolor='none', alpha=.8)

    bounds = date_termini[0][1].bounds
    for _,term in date_termini:
        bounds = (
            min(bounds[0],term.bounds[0]),
            min(bounds[1],term.bounds[1]),
            max(bounds[2],term.bounds[2]),
            max(bounds[3],term.bounds[3]))
    x0,y0,x1,y1 = bounds
    ax.set_extent(extents=(x0-5000,x1+5000,y0-5000,y1+5000), crs=mapinfo.crs)

    # Plot scale in km
    cartopyutil.add_osgb_scalebar(ax)
