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
from uafgi import bedmachine,glacier



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
