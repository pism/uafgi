import pandas as pd
import uafgi.data
from uafgi import pdutil,functional

@functional.memoize
def read(map_wkt):
    ##### Brief communication: Getting Greenland’s glaciers right – a new data set of all official Greenlandic glacier names
    # Bjørk, Kruse, Michaelsen, 2015 [BKM15]
    # RGI_ID = Randolph Glacier Inventory: not on GrIS
    # GLIMS_ID = Global Land Ice Measurements from Space: not on GrIS

    df = pd.read_csv(uafgi.data.join('GreenlandGlacierNames/tc-9-2215-2015-supplement.csv'))

    # Keep only glaciers on the ice sheet (285 of them)
#    df = df[df['Type'] == 'GrIS']
    #assert(len(df) == 285)

    # Drop columns not useful for GrIS glaciers
    df = df.drop(['RGI_ID', 'GLIMS_ID', 'RGI_LON', 'RGI_LAT', 'Type'], axis=1) \
        .rename(columns={'ID' : 'id', 'LAT' : 'lat', 'LON' : 'lon',
        'New_Greenl' : 'new_greenl_name', 'Old_Greenl' : 'old_greenl_name', 'Foreign_na' : 'foreign_name',
        'Official_n' : 'official_name', 'Alternativ' : 'alt'})

    return pdutil.ext_df(df, map_wkt,
        add_prefix='bkm15_',
        keycols=['id'],
        lonlat=('lon','lat'),
        namecols=['official_name', 'new_greenl_name', 'old_greenl_name', 'foreign_name', 'alt'])


