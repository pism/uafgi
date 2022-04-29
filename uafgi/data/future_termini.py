import functools
import pandas as pd
from uafgi import shputil,pdutil
import uafgi.data
import rtree

@functools.lru_cache()
def _read_bare(map_wkt):
    # Reads Elizabeth's hand-drawn future "termini" (flux gates)
    df = pd.DataFrame(shputil.read(uafgi.data.join('HypotheticalTermini', 'HypotheticalTermini.shp'), wkt=map_wkt)) \
        .drop(['_shape0', 'id'], axis=1) \
        .rename(columns={'_shape':'terminus'})

    # fid = Feature ID, to be used when reading shapefile
    df = df.reset_index().rename(columns={'index':'ftid'})    # Add a key column

    return pdutil.ext_df(df, map_wkt, add_prefix='ft_',
        units={},
        keycols=['ftid'])

def read(map_wkt):
    ft = _read_bare(map_wkt)
    fj = uafgi.data.fj.read(map_wkt)

    # Populate R-tree index with bounds of polygons
    print(fj.df.columns)
#    fj.df = fj.df.set_index('fj_fid')
    idx = rtree.index.Index()
    for fjord_ix,row in fj.df.iterrows():
        idx.insert(fjord_ix, row['fj_poly'].bounds)

    # Loop through terminus lines
    fids = list()
    print(ft.df.columns)
    for tix,trow in ft.df.iterrows():
        terminus = trow['ft_terminus']
        tbounds = terminus.bounds

        intersecting_fjords = list()
        for fjord_ix in idx.intersection(terminus.bounds):
            fj_poly = fj.df.loc[fjord_ix].fj_poly
            amount_intersect = fj_poly.intersection(terminus).length
            if amount_intersect > 0:
                intersecting_fjords.append((amount_intersect, fjord_ix))
        _,fjord_ix = max(intersecting_fjords)
        fids.append(fjord_ix)
    ft.df['fj_fid'] = fids

    return ft


