import pandas as pd
from uafgi import pathutil,gicollections
import pyproj
import shapely

def split_na(df, col):
    """Splits a dataframe by whether a value is missing in a column"""
    isna = df[col].isna()
    return df[~isna], df[isna]

# ----------------------------------------------------------------------
def points_col(lon_col, lat_col, transform_wgs84):
    """Creates a Series with projected Shapely points
    lon_col:
        DataFrame column with longitude
    lat_col:
        DataFrame column with latitude
    transform_wgs84:
        pyproj transformer from lon/lat to x/y
    """
    return pd.Series(
        index=lon_col.index,
        data=[shapely.geometry.Point(x,y) for x,y in zip(
            *transform_wgs84.transform(lon_col.tolist(), lat_col.tolist()))]
        )
# ----------------------------------------------------------------------
class ExtDf(gicollections.MutableNamedTuple):
    """A dataframe with a unit dict attached"""
    __slots__ = (
        'prefix',
        'df',
        'units',
        'keycols',
        'map_wkt')

    def __repr__(self):
        return 'ExtDf({})'.format(self.prefix)


#__slots__ = ('df', 'units', 'namecols')
def ext_df(df0, map_wkt, add_prefix=None, units=dict(), lonlat=None, namecols=None, keycols=None):

    # Save basic stuff
    self_units = units

    # ---------------------------------------------
    # Mess with projections
    self_map_wkt = map_wkt

    # wgs84: lon/lat from selections DataFrame (above)
    #        (ultimately from glacier location database files)
    wgs84 = pyproj.CRS.from_epsg("4326")

    # Assume all grids use the same projection (NSIDC Polar Stereographic)
    # Project everything to it
    map_crs = pyproj.CRS.from_string(map_wkt)

    # Transform from lat/lon into map projection
    transform_wgs84 = pyproj.Transformer.from_crs(wgs84,map_crs,always_xy=True)
    # -------------------------------------------------


    # Shallow copy
    self_df = df0.copy()

    
    # Figure out prefix stuff
    if add_prefix is None:
        # Infer common prefix in column names, if any
        self_prefix = pathutil.commonprefix(list(df))
    else:
        self_prefix = add_prefix

        # Add prefix to existing column names
        self_df.columns = [add_prefix + str(x) for x in self_df.columns]
        if namecols is not None:
            namecols = [self_prefix+x for x in namecols]
        if keycols is not None:
            keycols = [self_prefix+x for x in keycols]
        if lonlat is not None:
            lonlat = [self_prefix+x for x in lonlat]
        self_units = {self_prefix+name:val for name,val in self_units.items()}


    # Create a 'loc' column based on projecting the lon/lat coordinates of two original cols
    if lonlat is not None:
        self_df[self_prefix+'loc'] = points_col(self_df[lonlat[0]], self_df[lonlat[1]], transform_wgs84)
        self_units[self_prefix+'loc'] = 'm'    # TODO: Fish this out of the projection

    # -------------------------------------------------
    dropme = set()

    # Create a 'key' column
    if keycols is not None:
        if len(keycols) == 1:
            # Key column is just renamed other column
            self_df[self_prefix+'key'] = self_df[keycols[0]]
        else:
            # Put in a tuple
            self_df[self_prefix+'key'] = list(zip(*[self_df[x] for x in keycols]))
        self_keycols = keycols
#        if drop_cols:
#            dropme.update(self_keycols)
#            #self_df = self_df.drop(self_keycols, axis=1)
    else:
        self_keycols = list()

    # Make sure this is a proper unique key
    dups = self_df[self_prefix+'key'].duplicated(keep=False)
    dups = dups[dups]
    if len(dups) > 0:
        print('============================ Duplicate keys')
        print(self_df.loc[dups.index])

    # Create an 'allnames' column by zipping all columns indicating any kind of name
    if namecols is not None:
        self_df[self_prefix+'allnames'] = list(zip(*[self_df[x] for x in namecols]))
        self_namecols = namecols
#        if drop_cols:
#            dropme.update(self_namecols)
#            #self_df = self_df.drop(self_namecols, axis=1)
    else:
        self_namecols = list()

    self_df = self_df.drop(list(dropme), axis=1)

    return ExtDf(self_prefix, self_df, self_units, self_keycols, self_map_wkt)

# -------------------------------------------------------------
def check_dups(df, name, keycol, ignore_dups=False):
    """Checks for duplicate values of a column in a DataFrame.
    If dups are found, return the duplicated rows, sorted by the column.
    Or... it just removes the dups from the dataframe.

    df: DataFrame
    col: str
        Column to check for dups in
    """
    # Check for duplicate left keys
    dups0 = df[keycol].duplicated(keep=False)
    dups = dups0[dups0]

    # Remove all dups for testing...
    if ignore_dups:
        df = df[~dups0]
    else:
        if len(dups) > 0:
            ddf = pd.merge(df, dups, how='inner', left_index=True, right_index=True, suffixes=(None,'_DELETEME'))

            ddf = ddf.drop(
                [x for x in ddf.columns if x.endswith('_DELETEME')], axis=1)
            ddf = ddf.sort_values([keycol])

            print('ERROR:')
            print(ddf)
            ddf.to_csv('multi_err.csv')
            raise JoinError(
                'Multiple values of keycol {} found in {}'.format(keycol, name),
                ddf)
    return df
# -------------------------------------------------------------------------------
class JoinError(ValueError):
    def __init__(self, msg, df):
        super().__init__(msg)
        self.df = df

class Match(gicollections.MutableNamedTuple):
    __slots__ = (
        'xfs',     # (left, right): ExtDfs being joined
        'cols',    # Columns being joined in left and right
        'df')      # The match dataframe

    def __repr__(self):
        return 'Match({}, len={})'.format(
            str(list(zip(self.xfs, self.cols))),
            len(self.df))

    def swap(self):
        """Swaps left and right, returns new Match object"""
        return Match(self.xfs[::-1], self.cols[::-1], self.df)

    def left_join(self, overrides=None, ignore_dups=False, match_cols=None, right_cols=None):

        """Given a raw Match, returns a DataFrame that can be used for a left
        outer join.  It will have (assuming ExtDfs named 
          1. No more than copy of each row of the left ExtDf
          2. Columns: [<left>_ix, <left>_key, <right>_ix, <right>_key]

        Raises a JoinError if the (1) cannot be satisfied.

        match: Match
            Raw result of a match
        overrides: DataFrame
            Manual overrides for the join.
            Contains columns: [<left>_key, <right>_key]
        ignore_dups:
            Remove any duplicate rows (on the left key) from the join.
            Otherwise, raise a JoinError on any duplicate rows.
        match_cols: [str, ...]
            Columns from the match dataframe to keep in final result.
        right_cols: [str, ...]
            Names of columns from right DataFrame to keep.
            If None, keep all columns.
        """

        # Column names / shortcuts
        left = self.xfs[0]
        left_ix = left.prefix + 'ix'
        left_key = left.prefix + 'key'
        right = self.xfs[1]
        right_ix = right.prefix + 'ix'
        right_key = right.prefix + 'key'
#        print('left_ix={}, right_ix={}'.format(left_ix,right_ix))

        # Make dummy overrides
        if overrides is None:
            overrides = pd.DataFrame(columns=[left_ix, left_key, right_ix, right_key])

        else:
            # Add index columns to overrides (for convenient joining w/ other tables)
            for xf in self.xfs:
                df = xf.df
                key = xf.prefix+'key'

                try:
                    overrides = pd.merge(overrides, df[[key]].reset_index(), how='left', on=key) \
                        .rename(columns={'index':xf.prefix+'ix'})
                except KeyError as ke:
                    print('Is a column missing from your overrides table?')
                    raise

            # Keep only the required columns
            overrides = overrides[[left_ix, left_key, right_ix, right_key]]

            # Remove rows with missing right_key or left_key
            overrides = overrides.dropna(how='any', axis=0)

        # Make sure there are no duplicates in overrides
        overrides = check_dups(overrides, 'overrides', left_key)

        # Remove overridden rows (from left) from our match DataFrame
        df = pd.merge(self.df, overrides[[left_key]], how='left', on=left_key, indicator=True)
        df = df[df['_merge'] != 'both'].drop(['_merge'], axis=1)
        df = check_dups(df, left.prefix, left_key, ignore_dups=ignore_dups)

        # ----------------------------------------------------
        # Remove extraneous columns
        df = df[[left_ix, left_key, right_ix, right_key]]

        # Add in (manual) overrides (and re-index)
        if len(overrides) > 0:
            df = pd.concat([overrides,df], ignore_index=True)

        matchdf = df

        # -------------------------------------------
        # Do the join!
        # https://stackoverflow.com/questions/11976503/how-to-keep-index-when-using-pandas-merge

        # Join left -- matchdf
        df = left.df.copy()    # Shallow copy
        df['index'] = df.index
        df = pd.merge(df,
            matchdf, how='left',
            left_index=True, right_on=left_ix,
            suffixes=(None,'_DELETEME')).set_index('index')

        # Join (left -- matchdf) -- right
        df['index'] = df.index
        df = pd.merge(df,#.reset_index(),
            right.df if right_cols is None else right.df[right_cols],
            how='left', left_on=right_ix, right_index=True,
            suffixes=(None,'_DELETEME')).set_index('index')

#        df = df.reset_index()

#        try:
#            print('dddddddxxxxxxxxxxxxxxxxxxxxxxxxxf\n',df[['w21_key', 'cf20_key']])
#        except:
#            pass



        # Remove extra columns that accumulated in the join
        drops = {x for x in df.columns if x.endswith('_DELETEME')}
        drops.update([left_ix, right_ix])
        if (right_cols is not None) and (right_key not in right_cols):
            drops.add(right_key)
        df = df.drop(drops, axis=1)


        return self.xfs[0].replace(df=df)

def override_cols(df, overrides, keycol, cols):
    """Joins df with overrides on keycol; and replaces any of df.cols with overrides.cols

    keycol:
        Name of column to join on between df and overrides
    cols: [spec, ...]
        List of columns to match and override.
        Each spec specifies:
          * Name of column in df
          * Name of column in overrides
          * Name of column in resulting DataFrame
        Spec can be:

        col (str):
            Name of column in df, overrides and the result
        (df_col, over_col):
            Name of column in df and overrides.
            Resulting column will be named same as df_col
        (df_col, over_col, result_col):
            All three spelled out
    """

    # Ensure no dups in this (user-generated) table
    check_dups(overrides, 'overrides', keycol)


    # Translate the specs to most verbose form
    cols0 = cols
    cols = list()
    for spec in cols0:
        if type(spec) == str:
            cols.append((spec, spec, spec))
        elif type(spec) == tuple:
            if len(spec) == 2:
                cols.append((spec[0], spec[1], spec[0]))
            elif len(spec) == 3:
                cols.append(spec)
            else:
                raise TypeError(spec)
        else:
            raise TypeError(spec)

    ocols = [x[1] for x in cols]
    overrides = overrides[[keycol] + ocols] \
        .rename(columns=dict((x,x+'_OVERRIDE') for x in ocols))

    df = pd.merge(df, overrides, how='left', on=keycol)
    for df_col, over_col, result_col in cols:
        df[result_col] = df[over_col+'_OVERRIDE'].fillna(df[df_col])
        drops = [over_col+'_OVERRIDE']
        if result_col != df_col:
            drops.append(df_col)
        df = df.drop(drops, axis=1)

    return df

# ===========================================================================
def _point_poly_intersects(poly, threshold):
    def intersect_fn(p):    # p is Point or (Point, ...) or NaN
        if poly is None or (type(poly) == float and np.isnan(poly)):
            return False

        if type(p) == shapely.geometry.Point:
            return poly.intersects(p)
        elif type(p) == shapely.geometry.LineString:
            # Does it intersect
            return poly.intersects(p)
        elif type(p) == shapely.geometry.MultiPoint:
            # Proportion of points that overlap the polygon
            nintersect = 0
            for pp in p:
                if poly.intersects(pp):
                    nintersect += 1
            frac = (nintersect / len(p))
            return frac > threshold
#        elif type(p) == shapely.geometry.MultiLineString:
#            # Proportion of points that overlap the polygon
#            nintersect = 0
#            ntot = 0
##            coords = [x.coords for x in p]
##            print(coords)
##            print(len(coords))
#            types = [type(x) for x in p]
#            print('ttttttttt======================= ', types)
#            for ls in p:    # LineString
#                for xy in ls.coords:    # Point
#                    pp = shapely.geometry.Point(*xy)
#                    ntot += 1
#                    if poly.intersects(pp):
#                        nintersect += 1
#            frac = (float(nintersect) / ntot)
#            if nintersect>0:
#                print(nintersect, ntot)
#            return frac > threshold
        else:
            return False
    return intersect_fn

def _polys_overlapping_points(points_s, polys, poly_outs, poly_label='poly', threshold=.95):

    """Given a list of points and polygons... finds which polygons overlap
    each point.

    points_s: pd.Series(shapely.geometry.Point)
        Pandas Series containing the Points (with index)
    polys: [poly, ...]
        List of shapely.geometry.Polygon
    poly_outs: [x, ...]
        Item to place in resulting DataFrame in place of the Polygon.
        Could be the same as the polygon.

    """

    # Load the grids 
    out_s = list()
    for poly,poly_out in zip(polys,poly_outs):

        # Find intersections between terminus locations and this grid
        # NOTE: intersects includes selections.index
        ppi_fn = _point_poly_intersects(poly, threshold)
        intersects = points_s[points_s.map(ppi_fn)]
#        intersects = points_s[points_s.map(lambda p: poly.intersects(p) if type(p) == shapely.geometry.Point else False)]

        out_s.append(pd.Series(index=intersects.index, data=[poly_out] * len(intersects),name=poly_label))

    grids = pd.concat(out_s, axis=0)
    return grids

def match_point_poly(left, left_point, right, right_poly, left_cols=None, right_cols=None, threshold=.95):
    """Creates a Match object, looking for points in <left> contained in polygon in <right>
    left, right: ExtDf
        Datasets to match
    left_point: str
        Column in left to join.  Must be of type POINT
    right_poly: str
        Column in right to join.  Must be of type POLYGON
    left_cols, right_cols:
        Additional columns from left and right to include in the match dataframe
    """

    keyseries = _polys_overlapping_points(
        left.df[left_point], right.df[right_poly], right.df.index,
        threshold=threshold)
    keyseries.name = right.prefix+'ix'
    matchdf = keyseries.reset_index().rename(columns={'index':left.prefix+'ix'})

    # Add in extra cols
    for xcols,xf in zip((left_cols,right_cols), (left,right)):
        key = xf.prefix+'key'
        if xcols is None:
            xcols = [key]
        elif key not in xcols:
            xcols = [key] + xcols

        matchdf = pd.merge(matchdf, xf.df[xcols],
            how='left', left_on=xf.prefix+'ix', right_index=True)

    return Match((left,right), (left_point, right_poly), matchdf)

# ====================================================================
def merge_nodups(*args, **kwargs):
	"""An extension of pandas.merge that eliminates duplicates of columns with the same name.
	(It assumes columns with the same name are the same)."""
    df = pd.merge(*args, suffixes=(None,'_DELETEME'), **kwargs)
    drops = [x for x in df.columns if x.endswith('_DELETEME')]
    df = df.drop(drops, axis=1)
    return df
