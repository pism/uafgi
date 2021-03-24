import pandas as pd
import uafgi.data
from uafgi import pdutil,functional
import uafgi.data.greenland

@functional.memoize
def read(map_wkt):
    # Morlighem et al, 2017
    # BedMachine v3 paper

    colnames,units = uafgi.data.greenland.sep_colnames(
        ['id', 'name', 'lat', 'lon', ('speed', 'm a-1'), ('drainage', 'km^2'),
        'mc', 'bathy',
        ('ice_front_depth_b2013','m'), ('ice_front_depth_rtopo2','m'), ('ice_front_depth','m'),
        'contconn300_b2013', 'contcon300_rtopo2', 'contcon300',
        'contconn200_b2013', 'contcon200_rtopo2', 'contcon200',
        'bathy_coverage'])
    df = pd.read_csv(uafgi.data.join('morlighem2017/TableS1.csv'), skiprows=3,
        usecols=list(range(len(colnames))), names=colnames) \
        .drop('id', axis=1)
    df = df[df.name != 'TOTAL']

    return pdutil.ext_df(df, map_wkt,
        add_prefix='m17_',
        units=units,
        keycols=['name','lon','lat'],
        lonlat=('lon','lat'),
        namecols=['name'])
