import numpy as np
import pandas as pd
import shapely,shapely.geometry,shapely.affinity
from uafgi.util import shapelyutil

epsilon = 1e-14

class ScaledNorm:
    """Computes L2 norms for values in pixel coordinates, but outputs
    lengths in geographic coordinates"""
    def __init__(self, dx, dy):
        self.dx2 = dx*dx
        self.dy2 = dy*dy

    def __call__(self, x0,y0,x1,y1):
        xx = x1-x0
        yy = y1-y0
        return np.sqrt(xx * xx * self.dx2 + yy * yy * self.dy2)

def axis_crossings(x0, y0, x1, y1, xycols, norm_fn):

    """For an line segment in pixel-scaled space, determines where that
    segment crosses the x axis.  This can also be used to determine y
    axis crossings by flipping the x and y axes by the caller.

    x0, y0, x1, y1: float
        End points of the line segment
    xycols:
        Labels used to produce correctly labeled dataframe on output.
        ['x', 'y'] if we are determining x axis crossings
        ['y', 'x'] if we are determining y axis crossings
    norm_fn(x0,y0,x1,y1):
        Function determines the length of a line segment in geographic
        coordinates.  (See ScaledNorm above).

        NOTE: Like xycols, x and y need to be flipped when we are
              determining y axis crossings.

    Returns: DataFrame
        One row for each time the line segment crosses the axis.
        Does NOT include endpoints uniless they also intersect with the axis.

        t: float (geographic coordinates)
            Distance from (x0,y0) to the axis crossing
        x, y: float (pixel coordinates)
            Coordinates of the axis crossing (
    """


    # x coordinates crossed by the line
    x_crossings_x = np.arange(np.ceil(x0), x1 + epsilon)

    # (x-x0) = my (y - y0)
    m = (y1-y0) / (x1-x0)    # slope
    x_crossings_y = m * (x_crossings_x - x0) + y0

    # t = distance along the line
    x_crossings_t = norm_fn(x0, y0, x_crossings_x, x_crossings_y)

    df = pd.DataFrame({'t': x_crossings_t, xycols[0]: x_crossings_x, xycols[1]: x_crossings_y})
    return df


def _make_positive(nx, x0, x1):
    """The core algorithm only works for line segments with increasing
    x and y coordinates.  If either axis is decreasing, then this
    function will flip that coordinate to result in an increasing
    line.  After the algorithm finishes, values are flipped again to
    produce results in the original unflipped coordinates.

    This function is called once each for the x and y coordinate.

    x0, x1:
        x (or y) coordinate of beginning or end of line segment.
    """

    if x1 > x0:
        return x0, x1, False
    else:
        return nx-x0, nx-x1, True


def pixel_crossings(grid, x0, y0, x1, y1, drop_end=True):
    """Intersects a line segment with pixels, determining the (i,j)
    coordinate of each pixel traversed, and the length of line
    segment traversing that pixel.  NOTES:

        * The line segment x0,y0,x1,y1 is in PIXEL coordinates (not
          geographic)
        * Accounts 

    grid: gisutil.RasterInfo
        The grid being intersected with
    x0,y0,x1,y1: float
        The line segment in pixel coordinates (not geographic)
    drop_end: (default True)
        If True, drop the last point in the dataframe.
        This results in one line per line segment, but without
        explicit record of the ending point.

    Returns: DataFrame
        One row per crossing of a pixel boundary, sorted by distance from (x0,y0)
        If not drop_end, then there will also be a final row

        t: float
            Distance from (x0,y0)
        x,y: float
            Location of pixel boundary (in pixel coordinates, not geographic)
        i,j: int
            Coordinates of the line segment that starts with the pixel
            boundary in this row and ends at the pixel boundary in the
            next row.
        len: float
            Length of the line segment through pixel (i,j)
            If not drop_end, then this will be NaN for the last row.
    """
    # Ensure that x1>x0 and y1>y0
    x0, x1, flipx = _make_positive(grid.nx, x0, x1)
    y0, y1, flipy = _make_positive(grid.ny, y0, y1)


    # Collect x-crossings, y-crossings and beginning & end points, and
    # assemble them all into one dataframe.
    normxy = ScaledNorm(grid.dx, grid.dy)
    dfs = list()
    if x1 != x0:
        dfs.append(axis_crossings(x0, y0, x1, y1, ['x', 'y'], normxy))
    if y1 != y0:
        normyx = ScaledNorm(grid.dy, grid.dx)
        dfs.append(axis_crossings(y0, x0, y1, x1, ['y', 'x'], normyx))

    # Beginning and ending points
    dfs.append(pd.DataFrame([
        (0, x0, y0),
        (normxy(x0,y0,x1,y1), x1, y1),
        ], columns=['t', 'x', 'y']))

    df = pd.concat(dfs).sort_values(['t']).reset_index(drop=True)

    # Drop duplicates, identifiable because by having the same value
    # of t (distance from (x0,y0))
    df = df.groupby('t', as_index=False).agg({'x': 'first', 'y': 'first'})

    # Compute the length of each line segment by differencing
    # successive values of t
    df['len'] = -df.t.diff(periods=-1)

    # Un-flip x and y coordinates, if either was flipped.
    for flipp,xx,nxx,ii in ((flipx,'x',grid.nx,'i'), (flipy,'y',grid.ny,'j')):
        if flipp:
            df[xx] = df[xx].map(lambda x: nxx - x)
        df[ii] = df[xx].map(int)
        if flipp:
            # For flipped coordinates, the original (i,j) is the END
            # not the beginning.  So we have to shift by one.
             df[ii] = df[ii].shift(-1, fill_value=-1)

    # Remove end point if desired
    if drop_end:
        df = df.iloc[:-1]
    return df

# ------------------------------------------------------------------------------
def _iterate_Point(ls):
#    return
#    yield None
    yield from ()    # Yield nothing

def _iterate_LineString(ls):
    for p0,p1 in zip(ls.coords[:-1], ls.coords[1:]):
        yield p0[0], p0[1], p1[0], p1[1]
def _iterate_collection(ls):
    for part in ls.geoms:
        yield from iterate_ls(part)

_iterate_ls = {
    shapely.geometry.Point: _iterate_Point,
    shapely.geometry.LineString: _iterate_LineString,
    shapely.geometry.GeometryCollection: _iterate_collection,
    shapely.geometry.MultiLineString: _iterate_collection,
}


def iterate_ls(ls):
    """
    Iterates the line segments in a LineString or MultiLineString
    (This is a facade to iterate through either one)

    ls:
        LineString or MultiLineString
    yields:
        x0,y0,x1,y1: float
            Each line segment
    """
    try:
        xiter_fn = _iterate_ls[type(ls)]
    except KeyError:
        raise TypeError(f'I do not know how to iterate over {ls}')

    result = yield from xiter_fn(ls)
    return result



#    if isinstance(ls, shapely.geometry.LineString):
#    elif isinstance(ls, shapely.geometry.MultiLineString):
#        for part in ls.geoms:
#            for p0, p1 in zip(part.coords[:-1], part.coords[1:]):
#                yield p0[0], p0[1], p1[0], p1[1]
#    elif isinstance(ls, shapely.geometry.GeometryCollection):
#        for part in ls.geoms:
#    elif isinstance(ls, shapely.geometry.Point):
#        return
#    else:


def linestrings_crossings(lss, grid, index_col_name='lsix'):
    """Intersects an array of LineStrings or MultiLineStrings with
    pixels in a grid, determining the (i,j) coordinate of each pixel
    traversed, and the length of line segment traversing that pixel.

    lss: Pandas Series of Shapely LineStrings or MultiLineStrings
        This will typically be the the *geometry* column from a GeoDataFrame.
    grid:
        The grid defining the current tile (no margin)
    index_col_name:
        Name to use for the output column taken from the input index

    Returns: DataFrame
        One row per pixel crossing, i.e. many rows per item in lss

        lsix:
            Index value of the LineString in lss that is crossing a pixel
        t: float
            Distance from (x0,y0)
        x,y: float
            Location of pixel boundary (in pixel coordinates, not geographic)
        i,j: int
            Coordinates of the line segment that starts with the pixel
            boundary in this row and ends at the pixel boundary in the
            next row.
        len: float
            Length of the line segment through pixel (i,j)
            If not drop_end, then this will be NaN for the last row.
        

    """

    dfs = list()

    # Transform to pixel coordiantes (1 = 1 pixel width)
    geoinv_affine = shapelyutil.to_affine(grid.geoinv)
    bbox = shapely.affinity.affine_transform(grid.bounding_box(), geoinv_affine)
    bbox = shapely.orient_polygons(bbox, exterior_cw=False)     # Might be needed...

    # Clip to bounding box
#    print('--------- lss0')
#    print(lss)

    lss = lss.map(lambda ls: shapely.intersection(
        shapely.affinity.affine_transform(ls, geoinv_affine),
        bbox))

#    print('--------- lss1')
#    print(lss)

    # Find pixel crossings for each line segment
    for lsix,ls in lss.items():    # ix = Index value
#        print(f'lsix {lsix} of {len(lss)}')
        for xyxy in iterate_ls(ls):
            # Flip line segment to slope-positive
            df = pixel_crossings(grid, *xyxy)
            df[index_col_name] = lsix
            dfs.append(df)
#            print(df)

    if len(dfs) > 0:
        # Put it together into one giant dataframe
        df = pd.concat(dfs)
        return df

    return pd.DataFrame([], columns=[index_col_name, 't', 'x', 'y', 'i', 'j'])

    return None
# ---------------------------------------------------------------------
