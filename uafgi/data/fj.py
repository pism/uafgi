import pandas as pd
import uafgi.data
from uafgi import pdutil,functional,shputil

@functional.memoize
def read(map_wkt):
    # Reads Elizabeth's hand-drawn fjord outlines
    # They are constructed to each contain the point from w21
    # No labels.  Join an existing table, with locations obtained from bkm21,
    # to add fjord outlines to it.
    df = pd.DataFrame(shputil.read(uafgi.data.join('fj/fjord_outlines.shp'), map_wkt)) \
        .drop(['_shape0', 'id'], axis=1) \
        .rename(columns={'_shape':'poly'})

    # fid = Feature ID, to be used when reading shapefile
    df = df.reset_index().rename(columns={'index':'fid'})    # Add a key column

    return pdutil.ext_df(df, map_wkt, add_prefix='fj_',
        units={'fjord_poly': 'm'},
        keycols=['fid'])
