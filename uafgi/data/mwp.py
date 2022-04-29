import pandas as pd
import uafgi.data
from uafgi import pdutil,functional,shputil

@functional.memoize
def read_mwb(map_wkt):

    # mwb: Greenland Basins
    #If you want to use the basins in a publication, we will need to contact Micheal Wood
    # Name_AC is from Micheal, Name is from Eric Rignot, gl_type is the glacier type: TW: tide-water, LT: land terminating. UGID is a unique identifier that I use to identify glaciers in my scripts.
    # NOTE: mwb_basin == w21_coast

    df = pd.DataFrame(shputil.read('data/GreenlandBasins/Greenland_Basins_PS_v1.4.2c.shp', map_wkt)) \
        .drop('_shape0', axis=1) \
        .rename(columns={'basin':'basin', 'Name_AC':'name_ac', 'gl_type':'gl_type',
        'id':'id', 'id2':'id2', 'UGID':'ugid', 'Name':'name', '_shape':'basin_poly'})

    # Remove the "ICE_CAPS_NW" polygon
    df = df[df['name_ac'] != 'ICE_CAPS_NW']

    return pdutil.ext_df(df, map_wkt, add_prefix='mwb_',
        units={'basin_poly': 'm'},
        keycols=['name_ac'],
        namecols=['name_ac', 'name'])

