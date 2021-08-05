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
import uafgi.data.wkt
from uafgi.data import greenland,stability
import pickle


def select_glaciers():
    """Determine a set of glaciers for our experiment.
    Returns: A dataframe with many columns.
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

    return select
