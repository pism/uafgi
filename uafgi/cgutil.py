import scipy.stats
import numpy as np
import shapely.geometry
import shapely.ops

# General computational geometry (CG) utilities

def extend_linestring(gline0, xlen):
    """Give a linestring, extends it in the same general direction by a given amount.
    Does this by computing a least square fit for the angle; and then extending tby xlen
    length, at that angle, at both ends.

    gline0: LineString
        The original LineString
    xlen: float
        How far to extend in both directions
    """

    coords = list(gline0.coords)
    xcoords = [xy[0] for xy in coords]
    ycoords = [xy[1] for xy in coords]

    # Get least squares fit through the points
    slope, intercept, r_value, p_value, std_err = \
        scipy.stats.linregress(xcoords, ycoords)

    if abs(slope) > 1:
        slope, intercept, r_value, p_value, std_err = \
            scipy.stats.linregress(ycoords, xcoords)

        deltay = np.sign(slope) * np.sqrt(xlen*xlen / ((1.+slope)*(1.+slope)))

        return shapely.geometry.LineString(
            [(slope*(ycoords[0]-deltay) + intercept, ycoords[0]-deltay)] + \
            coords + \
            [(slope*(ycoords[-1]+deltay) + intercept, ycoords[-1]+deltay)])
    else:
        deltax = np.sign(slope) * np.sqrt(xlen*xlen / ((1.+slope)*(1.+slope)))

        return shapely.geometry.LineString(
            [(coords[0][0]-deltax, slope*(coords[0][0]-deltax) + intercept)] + \
            coords + \
            [(coords[-1][0]+deltax, slope*(coords[-1][0]+deltax) + intercept)])



