# https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
def moving_average(x, w):

    """w:
        Width of moving average (integer number of points).
        Assumes equispaced data."""
    return np.convolve(x, np.ones(w), 'valid') / w
