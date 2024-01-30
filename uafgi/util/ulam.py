import numpy as np

# Implements the Ulam Spiral
# It's a nice way to assign increasing 1D numbers to gridcells near a center cell.


def n_to_xy(n):
    """Ufunc"""
    sqrt_n = np.sqrt(n)
    m = int(np.floor(sqrt_n))
    if int(np.floor(2*sqrt_n))%2 == 0:
        k1x = n-m*(m+1)
        k1y = 0
    else:
        k1x = 0
        k1y = n-m*(m+1)

    sgn = int(pow(-1,m))
    k2 = int(np.ceil(.5*m))

    x = sgn * (k1x + k2)
    y = sgn * (k1y - k2)
    return (x,y)

def xy_to_n(xy):
    x,y = xy
    sgn = -1 if x<y else 1
    k = max(np.abs(x), np.abs(y))

    n = (4*k*k) + sgn * (2*k + x + y)

    return n

#n0 = np.arange(170)
#xy1 = [n_to_xy(x) for x in n0]
#print(xy1)
#n2 = [xy_to_n(z) for z in xy1]
#print(n2)
#assert n2 == list(n0)
