import scipy.interpolate
import numpy as np



# https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def spline_ma(xs, ys, dx_approx, ma_window):
    """
    xs0:
        Unevenly spaced x points
    ys0:
        Unevenly spaced y points
    dx_approx:
        Approx. resolution at which to subsample
    ma_window:
        Approx. size of moving average window (in coordinate space)
        to take the moving average

    Returns: scipy.interpolate.UnivariateSpline
        Valid from [xs[0]+ma_window, xs[-1]]
    """

    # Create a spline on all the knots (not smoothed)
    spl1 = scipy.interpolate.UnivariateSpline(xs, ys)

    # Resample to evenly spaced points
    range = xs[-1] - xs[0]
    nintervals = int(np.round(range / dx_approx))    # number of points = nintervals + 1
    dx = (xs[-1] - xs[0]) / nintervals
    xs1 = np.linspace(xs[0], xs[-1], nintervals)
    ys1 = spl1(xs1)
    print(len(xs1), len(ys1))

    # Run moving average
    ma_npt = int(np.round(ma_window / dx))
    print('ma_npt = ',ma_npt)
    ys2 = moving_average(ys1,ma_npt)
    xs2 = xs1[-len(ys2):]
    print(len(xs2), len(ys2))

    # Make a new spline on the resampled points
    spl2 = scipy.interpolate.UnivariateSpline(xs2,ys2)

    return spl2

