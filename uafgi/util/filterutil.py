import numpy as np
import scipy
import scipy.ndimage

sigma=2.0                  # standard deviation for Gaussian kernel
truncate=4.0               # truncate filter at this many sigmas

def nanfilter(U, sigma, truncate=4.0):
    """Gaussian filtering a image with Nan

    See: https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python

        A Gaussian filter which ignores NaNs in a given array U can be
        easily obtained by applying a standard Gaussian filter to two
        auxiliary arrays V and W and by taking the ratio of the two to
        get the result Z.

        Here, V is copy of the original U with NaNs replaced by zeros
        and W is an array of ones with zeros indicating the positions
        of NaNs in the original U.

        The idea is that replacing the NaNs by zeros introduces an
        error in the filtered array which can, however, be compensated
        by applying the same Gaussian filter to another auxiliary
        array and combining the two.

    U: nd.array(rank=2)
        Image to filter, with nan values in places
    sigma:
        standard deviation for Gaussian kernel
    truncate:
        truncate filter at this many sigmas

    """

    V=U.copy()
    V[np.isnan(U)]=0
    VV=scipy.ndimage.gaussian_filter(V,sigma=sigma,truncate=truncate)

    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=scipy.ndimage.gaussian_filter(W,sigma=sigma,truncate=truncate)

    Z=VV/WW
    return Z

