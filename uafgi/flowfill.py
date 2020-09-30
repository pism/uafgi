import netCDF4
import scipy.sparse.linalg
import scipy.sparse
import numpy as np
from pism.util import fill_missing_petsc
import uafgi.indexing
import scipy.ndimage
import math
from scipy import signal
from uafgi import make
from uafgi import ncutil, argutil
import sys

#np.set_printoptions(threshold=sys.maxsize)

# Enumerated values describing each gridcell
D_UNUSED = 0        # Not part of the domain
D_MISSING = 1       # Data are missing here
D_DATA = 2          # There are data here

# Indices and weights for first-order ceter fine difference.
center_diff = ((-1,1), (-.5,.5))

# -------------------------------------------------------
def get_indexing(ndarr):
    """Produces a uafgi.indexing.Indexing object describing
    how a standard row-major 2D array Numpy ndarray is indexed.

    ndarr:
        Numpy array for which to produce an Indexing object.
        Must be a standard row-major Numpy array with stride=1
    Returns:
        The Indexing object for ndarr
    """
    base = (0,0)
    extent = ndarr.shape
    indices = (0,1)    # List highest-stride index first
    return uafgi.indexing.Indexing(base, extent, indices)

# -------------------------------------------------------
def d_dy_present(divable,dy,  indexing,  rows,cols,vals, factor=1.0, rowoffset=0, coloffset=0):

    """Adds the discretized finite difference d/dy operator,
    (derivative in the y direction, or 0th index), to a sparse matrix.
    Each (j,i) index in the 2D array for which a derivative is being
    computed is converted to a k index in the 1D vector on which the
    matrix operates.

    The (row,col) of each item is based on the positions of the
    gridcells involved in each part of the d/dy operator.

    divable: ndarray(bool)
        Map of which cells are avaialble to calculate 2D derivatives;
        i.e. gridcells that are fully surrounded by gridcells with
        data (D_DATA), even if the gridcell itself is undefined.
        See get_divable()

    indexing: uafgi.indexing.Indexing
        Indexing object to convert between 2D and 1D indexing for the
        data arrays
    rows,cols,vals:
        Lists to which to append (row,col,val) for each item in
        the sparse matrix being created.
    factor:
        Multiple values by this
    rowoffset:
        Add this to every row index.
        Used to put a sub-matrix computing divergence, and one
        computing curl, into the same matrix.
    coloffset:
        Add this to every column index.
        Used to create a matrix that can take a concatenated vecotr of
        [v,u] as its input.
    """

    bydy = 1. / dy

    stcoo,stval = center_diff
    for jj in range(0, divable.shape[0]):
        for ii in range(0, divable.shape[1]):
            if divable[jj,ii]:
                for l in range(0,len(stcoo)):
                    jj2 = jj+stcoo[l]

                    # Convert to 1D indexing
                    k = indexing.tuple_to_index((jj,ii))
                    k2 = indexing.tuple_to_index((jj2,ii))
                    rows.append(rowoffset + k)
                    cols.append(coloffset + k2)
                    vals.append(factor * stval[l] * bydy)

def d_dx_present(divable,dx, indexing,  rows,cols,vals,
    factor=1.0, rowoffset=0, coloffset=0):
    """Adds the discretized finite difference d/dx operator,
    (derivative in the x direction, or 1st index), to a sparse matrix.
    See d_dx_present for arguments.

    d/dx is computed by computing d/dy on the transpose of all the
    (2D) array inputs.  The Indexing object must also be
    "tranposed"...
    """

    d_dy_present(np.transpose(divable),dx, indexing.transpose(),
        rows,cols,vals,
        factor=factor, rowoffset=rowoffset, coloffset=coloffset)
# ----------------------------------------------------------------
def div_matrix(d_dyx, divable, dyx, rows,cols,vals,
    factor=1.0, rowoffset=0):

    """Adds the discretized finite difference divergence operator to a
    sparse matrix.  Based on d/dy and d/dx functions.

    Matrix assumes concated vector of (v,u) where v is the velocity in
    the y direciton, and u in the x direction.

    Each (j,i) index in the 2D array for which a derivative is being
    computed is converted to a k index in the 1D vector on which the
    matrix operates.

    The (row,col) of each item is based on the positions of the
    gridcells involved in each part of the d/dy operator.

    d_dyx:
        Functions used to compute derivatives
        Must be (d_dy_present, d_dx_present)
    divable: ndarray(bool)
        Map of which cells are avaialble to calculate 2D derivatives;
        i.e. gridcells that are fully surrounded by gridcells with
        data (D_DATA), even if the gridcell itself is undefined.
        See get_divable()
    dyx: (dy, dx)
        Grid spacing in each direction
    rows,cols,vals:
        Lists to which to append (row,col,val) for each item in
        the sparse matrix being created.
    factor:
        Multiple values by this
    rowoffset:
        Add this to every row index.
        Used to put a sub-matrix computing divergence, and one
        computing curl, into the same matrix.

    """

    indexing = get_indexing(divable)
    n1 = divable.shape[0] * divable.shape[1]
    d_dyx[0](divable,dyx[0], indexing, rows,cols,vals,
        factor=factor, rowoffset=rowoffset)
    d_dyx[1](divable, dyx[1], indexing, rows,cols,vals,
        factor=factor, rowoffset=rowoffset, coloffset=n1)


def curl_matrix(d_dyx, divable, dyx, rows,cols,vals,
    factor=1.0, rowoffset=0):
    """Adds the discretized finite difference divergence operator to a
    sparse matrix.  Based on d/dy and d/dx functions.

    Arguments:
        Same as div_matrix()"""

    indexing = get_indexing(divable)

    n1 = divable.shape[0] * divable.shape[1]
    # curl = del x F = dF_y/dx - dF_x/dy
    d_dyx[1](divable, dyx[1], indexing, rows,cols,vals,
        factor=factor, rowoffset=rowoffset)
    d_dyx[0](divable, dyx[0], indexing, rows,cols,vals,
        factor=-factor, rowoffset=rowoffset, coloffset=n1)

# -------------------------------------------------------
def dc_matrix(d_dyx, divable, dyx, rows,cols,vals,
    factor=1.0):

    """Accumulates a matrix that converts the concatenated vector:
        [v, u]
    to the concatenated vector:
        [div, curl]

    d_dyx:
        Must be (d_dy_present, d_dx_present)
    divable: ndarray(bool)
        Map of which cells are avaialble to calculate 2D derivatives;
        i.e. gridcells that are fully surrounded by gridcells with
        data (D_DATA), even if the gridcell itself is undefined.
        See get_divable()
    dyx: (dy, dx)
        Grid spacing in each direction
    rows,cols,vals:
        Lists to which to append (row,col,val) for each item in
        the sparse matrix being created.
    factor:
        Multiple values by this

    """

    n1 = divable.shape[0] * divable.shape[1]

    div_matrix((d_dy_present, d_dx_present), divable, dyx, rows,cols,vals)
    curl_matrix((d_dy_present, d_dx_present), divable, dyx, rows,cols,vals, rowoffset=n1)

# -------------------------------------------------------
def get_divable(idomain2):

    """Returns a domain 2D ndarray (true/false) indicating where the
    divergence can be computed, using ONLY center differences.
    This will be points for which all four of its neighbors have data
    (but the central point doesn't necessarily have to have data).

    idomain2: ndarray(bool)
        Map of which points in the domain have data.
    """
    domain2 = np.zeros(idomain2.shape, dtype=bool)
    # Loop 1..n-1 to maintain a bezel around the edge
    for jj in range(1,idomain2.shape[0]-1):
        for ii in range(1,idomain2.shape[1]-1):
            domain2[jj,ii] = (idomain2[jj+1,ii] and idomain2[jj-1,ii] and idomain2[jj,ii+1] and idomain2[jj,ii-1])
    return domain2
# --------------------------------------------------------
def get_div_curl(vvel2, uvel2, divable_data2, dyx=(1.,1.)):
    """Computes divergence and curl of a (v,u) velocity field.

    vvel2: ndarray(j,i)
        y component of velocity field
    uvel2: ndarray(j,i)
        x component of velocity field
    divable_data2: ndarray(j,i, dtype=bool)
        Map of points in domain where to compute divergence and curl
        See get_divable()
    dyx:
        Size of gridcells in y and x dimensions.
        By default set to 1, because having div and cur scaled similarly to the
        original values works best to make a balanced LSQR matrix.

    Returns: (div, curl) ndarray(j,i)
        Returns divergence and curl, computed on the domain divable_data2
    """

    n1 = divable_data2.shape[0] * divable_data2.shape[1]

    # ------------ Create div matrix on DATA points
    rows = list()
    cols = list()
    vals = list()
    dc_matrix((d_dy_present, d_dx_present),
        divable_data2,
        dyx, rows,cols,vals)
    M = scipy.sparse.coo_matrix((vals, (rows,cols)),
        shape=(n1*2, n1*2))

    # ------------ Compute div/curl

    vu = np.zeros(n1*2)
    vu[:n1] = np.reshape(vvel2,-1)
    vu[n1:] = np.reshape(uvel2,-1)

    # ... in subspace
    divcurl = M * vu
    div2 = np.reshape(divcurl[:n1], divable_data2.shape)
    curl2 = np.reshape(divcurl[n1:], divable_data2.shape)

    div2[np.logical_not(divable_data2)] = np.nan
    curl2[np.logical_not(divable_data2)] = np.nan

    return div2,curl2
# ----------------------------------------------------------
def disc_stencil(radius, dyx):
    """Creates a disc-shaped 2D convolution stencil
    Size of the 2D stencil will be 2*radius x 2*radius

    radius: [m]
        Radius of the disc to create
    dyx: (dy,dx) [m]
        Grid spacing in y and x direction
    Returns:
        ndarray (float32) that is 1 inside a disc, and 0 elsewhere.
    """

    shape = tuple(math.ceil(radius*2. / dyx[i]) for i in range(0,2))
    st = np.zeros(shape, dtype='float32')
    for j in range(0,shape[0]):
        y  = (j+.5)*dyx[0]
        for i in range(0,shape[1]):
            x = (i+.5)*dyx[1]
            st[j,i] = (np.sqrt((y-radius)*(y-radius) + (x-radius)*(x-radius)) <= radius)

    return st

def get_trough(thk, bed, threshold, vsvel, usvel):
    """Returns a map (raster) of the main trough of the glacier.
    sqspeed:
        Ice speed^2
    """
    speed = np.hypot(vsvel*thk, usvel*thk)

    sx = scipy.ndimage.sobel(speed, axis=0)
    sy = scipy.ndimage.sobel(speed, axis=1)
    sob = np.hypot(sx,sy)

    # Look for highest 1% of values of speed change
    speedvals = speed.reshape(-1)
    speedvals = speedvals[~np.isnan(speedvals)]
    np.sort(speedvals)

    n = speedvals.shape[0]
    n //= 100
    threshold = np.mean(speedvals[-n:])

    trough = (speed > threshold)
    return trough

def get_dmap(has_data, thk, threshold, dist_channel, dist_front, dyx):
    """Creates a domain of gridcells within distance of cells in amount2
    that are >= threshold.

    has_data: (2D bool)
        Points that have U/V velocities available.
        has_data = np.logical_not(np.isnan(values))
    thk: (2D)
        Ice thickness
    threshold:
        Threshold to define edge of main channel by rapid changes in
        Sobel-filtered values of thk.
    dist_channel:
        Distance from the edge of the channel to include in domain
    dist_front:
        Distance from the glacier front to include in the domain
    dyx: (dy,dx)
        Grid spacing
    """


    with netCDF4.Dataset('thk.nc', 'w') as nc:
        nc.createDimension('y', thk.shape[0])
        nc.createDimension('x', thk.shape[1])
        nc.createVariable('thk', 'd', ('y','x'))[:] = thk


    # Sobel-filter the amount variable
    sx = scipy.ndimage.sobel(thk, axis=0)
    sy = scipy.ndimage.sobel(thk, axis=1)
    sob = np.hypot(sx,sy)

    # Get original domain, where thickness is changing rapidly
    domain0 = (sob > threshold).astype('float32')

    # Create a disc-shaped mask, used to convolve
    stencil = disc_stencil(dist_channel, dyx)
    print('stencil shape ',stencil.shape)

    # Create domain of points close to original data points
    domain = (signal.convolve2d(domain0, stencil, mode='same') != 0)
    if np.sum(np.sum(domain)) == 0:
        raise ValueError('Nothing found in the domain, something is wrong...')

    # Points close to the calving front
    # Get maximum value of Sobel fill.  This will be an ice cliff,
    # somewhere on the calving front.
    sobmax = np.max(sob)
    front = (sob >= .95*sobmax).astype('float32')
    fc = scipy.ndimage.measurements.center_of_mass(front)
    front_center = (fc[0]*dyx[0], fc[1]*dyx[1])

    # Create the dmap
#    print('n domain = {} {} has_data = {}'.format(np.sum(np.sum(domain0)), , np.sum(np.sum(has_data))))
    dmap = np.zeros(thk.shape, dtype='i') + D_UNUSED
    dmap[domain] = D_MISSING
    dmap[has_data] = D_DATA
    dmap[np.logical_not(domain)] = D_UNUSED

    # Focus on area near calving front
    dthresh = dist_front*dist_front
    for j in range(0,dmap.shape[0]):
        y  = (j+.5)*dyx[0]
        y2 = (y-front_center[0])*(y-front_center[0])
        for i in range(0,dmap.shape[1]):
            x = (i+.5)*dyx[1]
            x2 = (x-front_center[1])*(x-front_center[1])
            if y2+x2 > dthresh:
                dmap[j,i] = D_UNUSED


    return dmap

# ----------------------------------------------------------
def reduce_column_rank(cols):
    """Reduce the column rank of a sparse matrix by renumbering columns to
    eliminate empty columns.

    cols:
        Column of each element in the CSS-format version of the sparse matrix.

    Returns:
        cols_s:
            List of renumbered columns
        mvs_cols: list
            Convert renumbered columns back to originals.  i.e.:
                mvs_cols[cols_s[i]] == cols[i]
    """

    col_set = dict((c,None) for c in cols)    # Keep order
    #print('len(col_set) = {} -> {}'.format(len(cols), len(col_set)))
    mvs_cols = list(col_set.keys())
    svm_cols = dict((c,i) for i,c in enumerate(mvs_cols))
    cols_d = [svm_cols[c_s] for c_s in cols]
    print(cols_d[:100])
    return cols_d,mvs_cols

def reduce_row_rank(rows, bb):
    """Reduce the row rank of a sparse matrix by renumbering rows to
    eliminate empty rows.

    rows:
        Row of each element in the CSS-format version of the sparse matrix.
    bb:
        The right-hand side vector (as a list)

    Returns:
        rows_s:
            List of renumbered rows
        bb_d:
            Renumbered right-hand side
        mvs_rows: list
            Convert renumbered rows back to originals.  i.e.:
                mvs_rows[rows_s[i]] == rows[i]
    """

    row_set = dict((c,None) for c in rows)    # Keep order

    mvs_rows = list(row_set.keys())
    svm_rows = dict((c,i) for i,c in enumerate(mvs_rows))
    rows_d = [svm_rows[c_s] for c_s in rows]
#    bb_d = [svm_rows[c_s] for c_s in bb]
    bb_d = [bb[mvs_rows[i]] for i in range(0,len(mvs_rows))]
    return rows_d,bb_d,mvs_rows

# ----------------------------------------------------------
def fill_flow(vvel2, uvel2, dmap, clear_divergence=False, prior_weight=0.8):
    """Fills in missing values in a (theoretically divergence-free
    velocity field.

    vvel2, uvel2: ndarray(j,i)
        Volumetric flow fields (should have divergence=0 in theory)
    dmap: ndarray(j,i, dtype=int)
        Status of each gridcell.  See get_dmap()
        Gridcells set to D_MISSING will be filled in.
    clear_divergence:
        if True, zero out the divergence when filling.
        Otherwise, just Poisson-fill existing (non-zero) divergence.
    prior_weight: 0-1
        The amount to weight rows that pin values to the origianl data.

    Returns:
        vvel_filled, uvel_filled:
            Filled versions of vvel2, uvel2
        diagnostics: {label: ndarray(j,i)}
            Intermediate values, for inspection

    """

    diagnostics = dict()

    # ------------ Select subspace of gridcells on which to operate
    # Select cells with data
    divable_data2 = get_divable(dmap==D_DATA)
    indexing_data = get_indexing(divable_data2)

    # n1 = number of gridcells, even unused cells.
    # LSQR works OK with unused cells in its midst.
    n1 = dmap.shape[0] * dmap.shape[1]

    # -------------- Compute divergence and curl
    print('Computing divergence and curl')
    div2,curl2 = get_div_curl(vvel2, uvel2, divable_data2)
    diagnostics['div'] = div2
    diagnostics['curl'] = curl2

    # ---------- Apply Poisson Fill to div
    print('Applying Poisson fill')
    if clear_divergence:
        div2_f = np.zeros(dmap.shape)
    else:
        div2_m = np.ma.array(div2, mask=(np.isnan(div2)))
        div2_fv,_ = fill_missing_petsc.fill_missing(div2_m)
        div2_f = div2_fv[:].reshape(div2.shape)
        div2_f[:] = 0
    diagnostics['div_filled'] = div2_f

    # ---------- Apply Poisson Fill to curl
    curl2_m = np.ma.array(curl2, mask=(np.isnan(curl2)))
    curl2_fv,_ = fill_missing_petsc.fill_missing(curl2_m)
    curl2_f = curl2_fv[:].reshape(curl2.shape)
    diagnostics['curl_filled'] = curl2_f

    # ================================== Set up LSQR Problem
    rows = list()
    cols = list()
    vals = list()

    # --------- Setup domain to compute filled-in data EVERYWHERE
    # This keeps the edges of the domain as far as possible from places where
    # "the action" happens.  Edge effects can cause stippling problems.
    divable_used2 = (dmap != D_UNUSED)
    # Make a bezel around the edge
    divable_used2[0,:] = False
    divable_used2[-1,:] = False
    divable_used2[:,0] = False
    divable_used2[:,-1] = False

    # ------------ Create div+cov matrix on all domain points.
    # This ensures our solution has the correct divergence and curl
    # This will include some empty rows and columns in the LSQR
    # matrix.  That is not a problem for LSQR.
    print('Computing div-curl matrix to invert')
    dyx = (1.,1.)
    dc_matrix((d_dy_present, d_dx_present), divable_used2,
        dyx, rows,cols,vals)

    # ----------- Create dc vector in subspace as right hand side
    # ...based on our filled divergence and curl from above
    dc_s = np.zeros(n1*2)
    dc_s[:n1] = np.reshape(div2_f, -1)
    dc_s[n1:] = np.reshape(curl2_f, -1)
    bb = dc_s.tolist()    # bb = right-hand-side of LSQR problem

    # ------------ Add additional constraints for original data
    # This ensures our answer (almost) equals the original, where we
    # had data.
    # Larger --> Avoids changing original data, but more stippling
    print('Adding additonal constraints')
    for jj in range(0, divable_data2.shape[0]):
        for ii in range(0, divable_data2.shape[1]):

            if dmap[jj,ii] != D_DATA:
                continue

            # Exclude the main trough (fast-flowing ice) as boundary condition;
            # We want this to be consistent with the bed, so we will recompute
            if vvel2[jj,ii]*vvel2[jj,ii] + uvel2[jj,ii]*uvel2[jj,ii] > 1.e19:
                continue

            ku = indexing_data.tuple_to_index((jj,ii))
            try:
                rows.append(len(bb))
                cols.append(ku)
                vals.append(prior_weight*1.0)
                bb.append(prior_weight*vvel2[jj,ii])

                rows.append(len(bb))
                cols.append(n1 + ku)
                vals.append(prior_weight*1.0)
                bb.append(prior_weight*uvel2[jj,ii])

            except KeyError:    # It's not in sub_used
                pass

    # ================= Solve the LSQR Problem
    ncols_s = n1*2
    cols_d,mvs_cols = reduce_column_rank(cols)    # len(cols)==n1*2
    nrows_s = len(bb)
    rows_d,bb_d,mvs_rows = reduce_row_rank(rows, bb)

    # ---------- Convert to SciPy Sparse Matrix Format
    M = scipy.sparse.coo_matrix((vals, (rows_d,cols_d)),
        shape=(len(mvs_rows),len(mvs_cols))).tocsc()
    print('LSQR Matrix complete: shape={}, nnz={}'.format(M.shape, len(vals)))
    rhs = np.array(bb_d)

    # ----------- Solve for vu
    print('Solving LSQR')
    vu_d,istop,itn,r1norm,r2norm,anorm,acond,arnorm,xnorm,var = scipy.sparse.linalg.lsqr(M,rhs, damp=.0005)#, iter_lim=100)

    vu = np.zeros(ncols_s) + np.nan
    vu[mvs_cols] = vu_d

    # ----------- Convert back to 2D
    vv3 = np.reshape(vu[:n1], dmap.shape)
    uu3 = np.reshape(vu[n1:], dmap.shape)


    return vv3, uu3, diagnostics

def fill_surface_flow(vsvel2, usvel2, amount2, dmap, clear_divergence=False, prior_weight=0.8):
    """Fills in missing vlaues in a surface velocity field (not really
    divergence-free).

    vsvel2, usvel2: np.array(j,i)
        Surface velocities
    amount2:
        Multiply surface velocity by this to get volumetric velocity
        Generally, could be depth of ice.
        (whose divergence should be 0)
    dmap: ndarray(j,i, dtype=int)
        Status of each gridcell.  See get_dmap()
        Gridcells set to D_MISSING will be filled in.
    clear_divergence:
        if True, zero out the divergence when filling.
        Otherwise, just Poisson-fill existing (non-zero) divergence.
    prior_weight: 0-1
        The amount to weight rows that pin values to the origianl data.

    Returns: vvs3, uus3, diagnostics
        vvs3, uus3:
            Final filled and smothed surface velocities

        diagnostics {label: ndarray} is intermediate values:
        (vvel2, uvel2, vvel_filled, uvel_filled)

        vvel2,uvel2:
            Original non-filled volumetric velocities
        vvel_filled, uvel_filled:
            Filled volumetric velocities.
            Should have divergence=0
    """

    diagnostics = dict()

    # Get volumetric velocity from surface velocity
    vvel2 = vsvel2 * amount2
    uvel2 = usvel2 * amount2
    diagnostics['vvel'] = vvel2
    diagnostics['uvel'] = uvel2

    vvel_filled,uvel_filled,d2 = fill_flow(vvel2, uvel2, dmap, clear_divergence=clear_divergence, prior_weight=prior_weight)
    diagnostics.update(d2.items())
    diagnostics['vvel_filled'] = vvel_filled
    diagnostics['uvel_filled'] = uvel_filled

    # Convert back to surface velocity
    vvs3 = vvel_filled / amount2
    vvs3[amount2==0] = np.nan
    uus3 = uvel_filled / amount2
    uus3[amount2==0] = np.nan

    # Smooth: because our localized low-order FD approximation introduces
    # stippling, especially at boundaries
    # We need to smooth just over a single gridcell
    vvs3 = scipy.ndimage.gaussian_filter(vvs3, sigma=1.0)
    uus3 = scipy.ndimage.gaussian_filter(uus3, sigma=1.0)

    # Create pastiche of original + new
    if True:
        # Prefer new over original, to maintain continuity of flow in simulations
        missing2 = np.isnan(vvs3)
        vvs4 = np.copy(vvs3)
        vvs4[missing2] = usvel2[missing2]
        uus4 = np.copy(uus3)
        uus4[missing2] = usvel2[missing2]
    else:
        # Prefer original over new, to presere original data
        missing2 = np.isnan(vsvel2)
        vvs4 = np.copy(vsvel2)
        vvs4[missing2] = vvs3[missing2]
        uus4 = np.copy(usvel2)
        uus4[missing2] = uus3[missing2]


    return vvs4, uus4, diagnostics
# -------------------------------------------------------

class fill_surface_flow_rule(object):

    default_kwargs = dict(clear_divergence=True, prior_weight=0.8)

    def __init__(self, makefile, ipath, bedmachine_path, odir, max_timesteps=None, **kwargs0):
        self.max_timesteps = max_timesteps
        self.rule = makefile.add(self.run,
            (ipath,bedmachine_path),
#            (make.opath(ipath, odir, '_filled_fastice'),))
            (make.opath(ipath, odir, '_filled', replace='_pism'),))
        self.fill_kwargs = argutil.select_kwargs(kwargs0, self.default_kwargs)


    def run(self):
        # ------------ Read amount of ice (thickness)
        # 'outputs/bedmachine/W69.10N-thickness.nc'
#        print('Getting thk from {}'.format(self.rule.inputs[1]))
        with netCDF4.Dataset(self.rule.inputs[1]) as nc:
            thk2 = nc.variables['thickness'][:].astype(np.float64)
            bed2 = nc.variables['bed'][:].astype(np.float64)

        # Filter thickness, it's from a lower resolution
        thk2 = scipy.ndimage.gaussian_filter(thk2, sigma=2.0)

        # Amount is in units [kg m-2]
        rhoice = 918.    # [kg m-3]: Convert thickness from [m] to [kg m-2]
        amount2 = thk2 * rhoice


        # ========================= Read Data from Input Files
        # --------- Read uvel and vvel
        t = 0    # Time
        # 'outputs/velocity/TSX_W69.10N_2008_2020_pism.nc'
        with netCDF4.Dataset(self.rule.inputs[0]) as nc:

            # Create the output file by copying the structure of the input file
            with netCDF4.Dataset(self.rule.outputs[0], 'w') as ncout:

                cnc = ncutil.copy_nc(nc, ncout)
                cnc.createDimension('time', size=0)    # Unlimited
#                var_pairs = dict((x,x) for x in nc.variables.keys())
#                del var_pairs['v_ssa_bc']
#                del var_pairs['u_ssa_bc']
#                cnc.define_vars(var_pairs.items())
                cnc.define_all_vars(zlib=True)
                cnc.createVariable('thickness', 'i', ('y','x'), zlib=True)
                cnc.createVariable('bed', 'i', ('y','x'), zlib=True)
                cnc.createVariable('dmap', 'i', ('y','x'), zlib=True)
                cnc.createVariable('trough', 'i', ('y','x'), zlib=True)
                for vname in ('x', 'y', 'time', 'time_bnds'):
                    cnc.copy_var(vname, vname)

                ncout.variables['thickness'][:] = thk2
                ncout.variables['bed'][:] = bed2

            # Now process / copy the data
            for t in range(0,len(nc.dimensions['time'])):

                if (self.max_timesteps is not None) and (t >= self.max_timesteps):
                    break

                print('============== Timestep t={}'.format(t))

                nc_vvel = nc.variables['v_ssa_bc']
                nc_vvel.set_auto_mask(False)
                vsvel2 = nc_vvel[t,:].astype(np.float64)
                vsvel2[vsvel2 == nc_vvel._FillValue] = np.nan

                nc_uvel = nc.variables['u_ssa_bc']
                nc_uvel.set_auto_mask(False)    # Don't use masked arrays
                usvel2 = nc_uvel[t,:].astype(np.float64)
                usvel2[usvel2 == nc_uvel._FillValue] = np.nan

                print('Fill Value {}'.format(nc_uvel._FillValue))


                # ------------ Set up the domain map (classify gridcells)
                has_data = np.logical_not(np.isnan(vsvel2))
                edge_threshold = 300.
                dmap = get_dmap(has_data, thk=thk2, threshold=edge_threshold,
                    dist_channel=3000., dist_front=20000., dyx=(100.,100.))
                trough = get_trough(thk2, bed2, edge_threshold, vsvel2, usvel2)


                # Should be the same on all timesteps
                if t == 0:
                    with netCDF4.Dataset(self.rule.outputs[0], 'a') as ncout:
                        ncout.variables['dmap'][:] = dmap
                        ncout.variables['trough'][:] = trough

            #    with netCDF4.Dataset('dmap.nc', 'w') as nc:
            #        nc.createDimension('y', vsvel2.shape[0])
            #        nc.createDimension('x', vsvel2.shape[1])
            #        nc.createVariable('amount', 'd', ('y','x'))[:] = amount2
            #        nc.createVariable('dmap', 'd', ('y','x'))[:] = dmap

                # ----------- Store it
                vv3,uu3,diagnostics = fill_surface_flow(vsvel2, usvel2, amount2, dmap,
                    **self.fill_kwargs)  # clear_divergence=True, prior_weight=0.8
                diagnostics['thk'] = thk2
                diagnostics['dmap'] = dmap

                with netCDF4.Dataset(self.rule.outputs[0], 'a') as ncout:

                    ncout.variables['v_ssa_bc'][t,:] = vv3
                    ncout.variables['u_ssa_bc'][t,:] = uu3


                    # ----------- Store it (debugging)
                if False:
                    with netCDF4.Dataset('flowfill.nc', 'w') as nc:
                        nc.createDimension('y', vsvel2.shape[0])
                        nc.createDimension('x', vsvel2.shape[1])
                        nc.createVariable('vsvel', 'd', ('y','x'))[:] = vsvel2
                        nc.createVariable('usvel', 'd', ('y','x'))[:] = usvel2
                        nc.createVariable('amount', 'd', ('y','x'))[:] = amount2

                        nc.createVariable('vsvel_filled', 'd', ('y','x'))[:] = vv3
                        nc.createVariable('usvel_filled', 'd', ('y','x'))[:] = uu3

                        nc.createVariable('vsvel_diff', 'd', ('y','x'))[:] = vv3-vsvel2
                        nc.createVariable('usvel_diff', 'd', ('y','x'))[:] = uu3-usvel2


                        vvel2=diagnostics['vvel']
                        uvel2=diagnostics['uvel']

                        for vname,val in diagnostics.items():
                            nc.createVariable(vname, 'd', ('y','x'))[:] = val

                        fastice = (vvel2*vvel2 + uvel2*uvel2)
                        fastice[np.isnan(fastice)] = 1.e20
                        fastice = (fastice > 1.e19)
                        nc.createVariable('fastice', 'i', ('y','x'))[:] = fastice

                        dmap = diagnostics['dmap']
                        dmap[fastice] = 0
                        nc.createVariable('dmap_fastice', 'i', ('y','x'))[:] = dmap

                        nc.createVariable('trough', 'i', ('y','x'))[:] = trough


                    sys.exit(0)


