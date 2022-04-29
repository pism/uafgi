import pandas as pd
import uafgi.data
from uafgi import pdutil,functional,shputil
import shapely
from osgeo import ogr

@functional.memoize
def read(map_wkt):
    # Reads Elizabeth's hand-drawn fjord outlines
    # They are constructed to each contain the point from w21
    # No labels.  Join an existing table, with locations obtained from bkm21,
    # to add fjord outlines to it.
    df = pd.DataFrame(shputil.read(uafgi.data.join('fj/fjord_outlines.shp'), wkt=map_wkt)) \
        .drop(['_shape0', 'id'], axis=1) \
        .rename(columns={'_shape':'poly'})

    # fid = Feature ID, to be used when reading shapefile
    df = df.reset_index().rename(columns={'index':'fid'})    # Add a key column

    return pdutil.ext_df(df, map_wkt, add_prefix='fj_',
        units={'fjord_poly': 'm'},
        keycols=['fid'])


def match_to_points(glaciers, key_col, points_col, fj, debug_shapefile=None):
    """glaciers: ExtDf
        A dataframe full of glaciers from some source
    key_col:
        A unique key column in glaciers DF
    points_col:
        Name of a column in select containing a bunch of points; eg. from termini
            
    """

    ret = dict()

    # Get a column of the centroids of each point
    tloc_col = glaciers.df[points_col].map(lambda pts: shapely.geometry.MultiPoint(pts).centroid)
    tloc_name = glaciers.prefix + 'tloc'
    glaciers2 = glaciers.copy()
    glaciers2.df[tloc_name] = tloc_col

    # match points to the fjord polygons
    match = pdutil.match_point_poly(glaciers2, tloc_name, fj, 'fj_poly')
    select = match.left_join(right_cols=['fj_poly', 'fj_fid'])
    ret['select'] = select

    # Find duplicate matches
    # ...for fjords
    df = select.df.dropna()
    df.sort_values(['fj_fid'])
    dfdup = df[df.duplicated(subset=['fj_fid'], keep=False)].sort_values('fj_fid')
    fjdup = dfdup['fj_fid'].drop_duplicates()
    fjdup = pd.merge(fj.df, fjdup, on='fj_fid')
    fjdup = fj.replace(df=fjdup)
    ret['fjdup'] = fjdup

    # Write out duplicate shapes to debug_shapefile (if specified)
    if debug_shapefile is not None:
        fields = fields=[ogr.FieldDefn('fj_key',ogr.OFTInteger)]
        shputil.write_shapefile2(fjdup.df['fj_poly'].tolist(), debug_shapefile, fields, attrss=list(zip(fjdup.df['fj_key'].tolist())))

    # ...for glaciers
    glaciers2dup = pdutil.merge_nodups(glaciers2.df,dfdup[key_col], left_on=key_col, right_on=key_col)
    glaciers2dup = glaciers2dup[key_col].drop_duplicates()
    glaciers2dup = pd.merge(glaciers2.df, glaciers2dup, on=key_col)
    glaciers2dup = glaciers2.replace(df=glaciers2dup)
    ret['glaciersdup'] = glaciers2dup

    # Try to re-do the matches, this time with multi-points
    if len(fjdup.df) > 0 and len(glaciers2dup.df) > 0:
        ret['match2'] = pdutil.match_point_poly(glaciers2dup, points_col, fjdup, 'fj_poly')

    return ret

42
