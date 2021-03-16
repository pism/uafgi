import os,csv,re,datetime,itertools
import numpy as np
import pandas as pd
import shapely
import uafgi.data
from uafgi import pdutil,functional,shputil

def _parse_daterange(srange):
    return tuple(datetime.datetime.strptime(x,'%d%b%Y') for x in srange.split('-'))

@functional.memoize
def read(map_wkt):
    """Reads annual terminus lines"""

    ddir = uafgi.data.join('measures-nsidc0642')
    terminiRE = re.compile(r'termini_(\d\d)(\d\d)_v01\.2\.shp')
    dfs = list()
    leaves = [list(), list()]    # Format changes between 2012/13 and 2014/15
    for leaf in os.listdir(ddir):
        match = terminiRE.match(leaf)
        if match is not None:
            y0 = 2000+int(match.group(1))
            y1 = 2000+int(match.group(2))
            leaves[0 if y0<2014 else 1].append((leaf,y0,y1))

    all_namecols = ['GlacName', 'GrnlndcNam', 'Name','Official_n', 'AltName', 'GrnlndcNam']
    for leaf,y0,y1 in sorted(leaves[0]):
        df = pd.DataFrame(shputil.read(os.path.join(ddir,leaf), map_wkt))
        df['year0'] = y0
        df['year1'] = y1
        namecols = [x for x in all_namecols if x in df.columns]
        df['allnames'] = list(zip(*[df[x] for x in (namecols)]))
        df['date'] = df['DateRange'].map(_parse_daterange)
        df = df.drop(['_shape0', 'DateRange'] + namecols, axis=1)
        dfs.append(df)

    # Different data format 2014 and beyond
    for leaf,y0,y1 in sorted(leaves[1]):
        df = pd.DataFrame(shputil.read(os.path.join(ddir,leaf), map_wkt))
        df['year0'] = y0
        df['year1'] = y1
        namecols = [x for x in all_namecols if x in df.columns]
        df['allnames'] = list(zip(*[df[x] for x in (namecols)]))
        df['date'] = df['DATE'].map(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))
        df = df.drop(['_shape0', 'DATE'] + namecols, axis=1)
        dfs.append(df)


    df = pd.concat(dfs)
    df = df.reset_index(drop=True) \
        .rename(columns={'_shape': 'terminus'}) \
        .drop('Id', axis=1)

    return pdutil.ext_df(df, map_wkt,
        add_prefix='ns642_',
        keycols=['GlacierID', 'year1'])


def by_glacier_id(ns642):
    """Collects rows from original ns642 DataFrame by GlacierID.
    Breaks the terminus lines apart into multiple points."""

    dfg = ns642.df.groupby(by='ns642_GlacierID')

    data = list()
    for name, gr in dfg:
        pointss = [list(ls.coords) for ls in gr['ns642_terminus']]
        points = list(itertools.chain.from_iterable(pointss))    # Join to a single list
        data.append([name, shapely.geometry.MultiPoint(points)])
        
    df2 = pd.DataFrame(data=data, columns=['ns642_GlacierID', 'ns642_points'])
    df2['ns642_key'] = df2['ns642_GlacierID']
    xdf = ns642.replace(df=df2, keycols=['ns642_GlacierID'])
    return xdf
