import scipy.stats
import numpy as np
import shapely.geometry
import shapely.ops

# General computational geometry (CG) utilities

def fitp(xcoords, ycoords):
    """Fits a line through the endpoints"""
    # https://www.emathhelp.net/calculators/algebra-1/slope-intercept-form-calculator-with-two-points/
    y2 = ycoords[-1]
    y1 = ycoords[0]
    x2 = xcoords[-1]
    x1 = xcoords[0]
    m = (y2-y1) / (x2-x1)
    b = y2 - m*x2
    return m,b

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

#    # Get least squares fit through the points
#    slopex, interceptx, r_valuex, p_valuex, std_errx = \
#        scipy.stats.linregress(xcoords, ycoords)
#
#    slopey, intercepty, r_valuey, p_valuey, std_erry = \
#        scipy.stats.linregress(ycoords, xcoords)
#


    slopex, interceptx = fitp(xcoords, ycoords)
    slopey, intercepty = fitp(ycoords, xcoords)


    if abs(slopex) > 1:
#    if r_valuey > r_valuex:    # y is a better fit than x

        if ycoords[0] > ycoords[-1]:
            coords.reverse()


        deltay = np.sqrt(xlen*xlen / ((1.+slopey)*(1.+slopey)))

        return shapely.geometry.LineString(
            [(slopey*(coords[0][1]-deltay) + intercepty, coords[0][1]-deltay)] + \
            coords + \
            [(slopey*(coords[-1][1]+deltay) + intercepty, coords[-1][1]+deltay)])
    else:
        if xcoords[0] > xcoords[-1]:
            coords.reverse()


        deltax = np.sqrt(xlen*xlen / ((1.+slopex)*(1.+slopex)))

        return shapely.geometry.LineString(
            [(coords[0][0]-deltax, slopex*(coords[0][0]-deltax) + interceptx)] + \
            coords + \
            [(coords[-1][0]+deltax, slopex*(coords[-1][0]+deltax) + interceptx)])



