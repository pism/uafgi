import scipy.io
import uafgi.data
from uafgi import functional,matlabutil,pdutil
import pandas as pd
import numpy as np
from uafgi import immutarray

"""Data from the paper:

**Estimating Greenland tidewater glacier retreat driven by submarine melting**

Donald A. Slater1, Fiamma Straneo1, Denis Felikson2, Christopher M. Little3, Heiko Goelzer4,5, Xavier Fettweis6, and James Holte1

The Cryosphere, 13, 2489–2509, 2019
https://doi.org/10.5194/tc-13-2489-2019
"""

@functional.memoize
def read(map_wkt):

    matfile = scipy.io.loadmat(uafgi.data.join('slater2019', 'glaciers.mat'))
    slatlist = matlabutil.structured_to_dict(matfile['glaciers'], unnest=True, compound='_')

    # Convert lists to np.array timeseries
    for row in slatlist:
        for name,val in row.items():
            if isinstance(val, list):
                if len(val) == 0:
                    row[name] = None
                else:
                    row[name] = immutarray.ImmutArray(val)

    df = pd.DataFrame(slatlist)

    return pdutil.ext_df(df, map_wkt, add_prefix='sl19_',
        keycols=['rignotid'],
        lonlat=('lon','lat'),
        namecols=['rignotname'])

