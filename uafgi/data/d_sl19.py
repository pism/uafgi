import uafgi.data

"""Data from the paper:

**Estimating Greenland tidewater glacier retreat driven by submarine melting**

Donald A. Slater1, Fiamma Straneo1, Denis Felikson2, Christopher M. Little3, Heiko Goelzer4,5, Xavier Fettweis6, and James Holte1

The Cryosphere, 13, 2489–2509, 2019
https://doi.org/10.5194/tc-13-2489-2019
"""

@functional.memoize
def read(map_wkt):

    matfile = scipy.io.loadmat(uafgi.data.join('slater2019', 'glaciers.mat')
    slatdf = matlabutil.structured_to_dict(matfile['glaciers')

    return slatdf
