import numpy as np
import scipy.signal


#@article{sobel19683x3,
#  title={A 3x3 isotropic gradient operator for image processing (1968)},
#  author={Sobel, Irwin and Feldman, Gary and others},
#  journal={a talk at the Stanford Artificial Intelligence Project},
#  year={1968}
#}

#History and Definition of the so-called "Sobel Operator",
#more appropriately named the
#Sobel-Feldman Operator
#by Irwin Sobel
#February 2, 2014
#Updated June 14 2015
#https://www.researchgate.net/profile/Irwin-Sobel/publication/285159837_A_33_isotropic_gradient_operator_for_image_processing/links/5af73f41aca2720af9cf6063/A-33-isotropic-gradient-operator-for-image-processing.pdf


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

    slope2 = Hx*Hx + Hy*Hy
    slope = np.sqrt(slope2)
    return slope, np.divide(Tx*Hx + Ty*Hy, slope2)
