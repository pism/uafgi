import re,os
import pandas as pd
import shapely
import uafgi.data
from uafgi import pdutil,functional,shputil
import pyproj

@functional.memoize
def read(map_wkt):

    # Convert WKT to CRS
    wgs84 = pyproj.CRS.from_epsg("4326")
    crs1 = pyproj.CRS.from_string(map_wkt)
    transform = pyproj.Transformer.from_crs(wgs84, crs1, always_xy=True)

    # Reads the CALFIN stuff
    rows = list()
    ddir = uafgi.data.join('calfin/domain-termini')
    calfinRE = re.compile(r'(.*)_(.*)_(.*)_v(.*).shp')
    for leaf in os.listdir(ddir):
        match = calfinRE.match(leaf)
        if match is None:
            continue

        fname = os.path.join(ddir,leaf)

        df = pd.DataFrame(shputil.read(fname, map_wkt, read_shape=False))

        # All the rows are the same, except different terminus line / position / etc
        row = df.loc[0].to_dict()
        row['fname'] = fname
        row['uniqename'] = match.group(3).replace('-',' ')

        # Discard columns that aren't constant through the entire shapefile
        for col in ('Center_X', 'Center_Y', 'Latitude', 'Longitude', 'QualFlag', 'Satellite', 'Date', 'ImageID', 'Author'):
            del row[col]

        # Take the center point of each calving front, combine to a MultiPoint object
        loc = pdutil.points_col(df['Longitude'], df['Latitude'], transform)
        mp = shapely.geometry.MultiPoint(points=loc.to_list())
        row['locs'] = mp

        # Add this as a row for our final dataframe
        rows.append(row)

    df = pd.DataFrame(rows) \
        .rename(columns={'GlacierID':'glacier_id', 'GrnlndcN':'greenlandic_name', 'OfficialN':'official_name', 'AltName':'alt_name', 'RefName':'ref_name'})


    return pdutil.ext_df(df, map_wkt, add_prefix='cf20_',
        units={'basin_poly': 'm'},
        keycols=['uniqename'],
        namecols=['greenlandic_name', 'official_name', 'alt_name', 'ref_name'])
