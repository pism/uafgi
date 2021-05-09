import pandas_ods_reader
import pandas as pd
import uafgi.data
from uafgi import shputil

def _csv_to_tuple(val):
    """Converts key column value from comma-separated format to standard
    format."""

    if (type(val) == float) or (val is None):
        return val

    parts = val.split(',')
    if len(parts) == 1:
        return parts[0]
    return tuple(parts)

def _read_overrides(overrides_ods, locations_shp, keycols, join_col, map_wkt):

    """Reads an per-project overrides table, ready to use as overrides in
    joins.  The table is read as a combination of an ODS file and a
    shapefile containing the locations of termini; with an attribute
    in the shapefile matching a column in the overrides_ods file.

    The location of the terminus points is returned in the columns:
        lat, lon: degrees
        loc: shapely.geometry.Point

    Args:
        overrides_ods: filename
            Name of the overrides file (ODS format)
            Must contain at least column: <join_col>

        locations_shp: filename
            Name of the shapefile identifying a location point for each glacier.
            Must contain at least one attribute: <join_col>

        keycols: [str, ...]
            Names of columns that are keys (in both datasources).
            Comma-separate them and turn into tuples.

        join_col: str
            Name of column used to join overrides and locations

    """

    # Manual overrides spreadsheet
    over = pandas_ods_reader.read_ods(overrides_ods,1)
    over = over.drop('comment', axis=1)
    over = over.dropna(how='all')    # Remove blank lines

    # Manual glacier point locations
    tl = pd.DataFrame(shputil.read(locations_shp, wkt=map_wkt))

    # Convert keycols to tuples
    for df in (over,tl):
        for col in keycols:
            df[col] = df[col].map(_csv_to_tuple)

    # Mark to remove any "extra" columns that we didn't actually join with
    df = pd.merge(over,tl,how='left',on='w21_key', suffixes=(None,'_DELETEME'))


    # Move data from the shapefile to override the lon/lat columns
    df = df.rename(columns={'_shape' : 'loc'})
    lon = df['_shape0'].map(lambda xy: xy if type(xy)==float else xy.x)
    lat = df['_shape0'].map(lambda xy: xy if type(xy)==float else xy.x)

    # Merge the shapefile and spreadsheet lat/lon, if available.
    if 'lon' in df:
        df['lon'] = df['_shape0'].map(lambda xy: xy if type(xy)==float else xy.x).fillna(df['lon'])
    if 'lat' in df:
        df['lat'] = df['_shape0'].map(lambda xy: xy if type(xy)==float else xy.y).fillna(df['lat'])

    # Remove extraneous columns
    drops = ['_shape0'] + [x for x in df.columns if x.endswith('_DELETEME')]
    df = df.drop(drops, axis=1)
    return df

def read_overrides():
    over = _read_overrides(
        uafgi.data.join('stability_overrides/overrides.ods'),
        uafgi.data.join('stability_overrides/terminus_locations.shp'),
        ['w21_key', 'bkm15_key'], 'w21_key', uafgi.data.wkt.nsidc_ps_north)
    return over