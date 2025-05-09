import numpy as np
import scipy.signal

# eq 3
Ox = np.array([
    [-1, -2, 0, 1, 2],
    [-4, -8, 0, 8, 4],
    [-6, -12, 0, 12, 6],
    [-4, -8, 0, 8, 4],
    [-1, -2, 0, 1, 2],
    ], dtype='d')
Oy = np.transpose(Ox)

def grad(val, dy, dx):
    Ty = scipy.signal.convolve2d(val, Oy, boundary='symm', mode='same') * (1. / (960. * dy))
    Tx = scipy.signal.convolve2d(val, Ox, boundary='symm', mode='same') * (1. / (960. * dx))
    return Ty,Tx



def compute_lapse(H, T, dy, dx):
    """Compute a gridded lapse rate based on local finite differences
    H: [m]
        Elevation
    T: [m]
        A gridded value
        (in this case, units of sx3 = [mm])
    dx,dy:
        Size of gridcell
    Returns: Units: [H]/[T] = [mm / m] == [m / km]
    """

    # A New Methodology for Estimating the Surface Temperature Lapse
    # Rate Based on Grid Data and Its Application in China

    # Compute gradient of elevations and values
    Hy,Hx = grad(H, dy, dx)
    Ty,Tx = grad(T, dy, dx)

    return Hx*Hx + H*Hy

    return np.divide(
        Tx*Hx + Ty*Hy,
        Hx*Hx + Hy*Hy)

