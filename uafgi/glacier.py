import numpy as np
import netCDF4
import os,subprocess
from cdo import Cdo
from uafgi import nsidc,cgutil,gdalutil,shapelyutil
from uafgi import giutil,cdoutil,make,ioutil,gicollections
import pandas as pd
import skimage.segmentation
import findiff

# Fjord pixel classifcation types
UNUSED = 0
LOWER_FJORD = 1
TERMINUS = 2
TERMINUS_EXTRA = 3    # Terminus line, outside of fjord
UPPER_FJORD = 4
UPPER_FJORD_SEED = 5

# Translate fjord classifications into lower / upper fjord;
# either including or not including the terminus line.
# LT = Less than (as in FORTRAN)
# LE = Less than or equal
# GE = Greater than or equal
# GT = Greater than
# Use:
#    fjc = classify_fjord(...)
#    lower_fjord = np.isin(fjc, glacier.LT_TERMINUS)
#    upper_fjord = np.isn(fjc, glacier.GE_TERMINUS)
LT_TERMINUS = (LOWER_FJORD,)
LE_TERMINUS = (LOWER_FJORD, TERMINUS)
GE_TERMINUS = (TERMINUS, UPPER_FJORD, UPPER_FJORD_SEED)
GT_TERMINUS = (UPPER_FJORD, UPPER_FJORD_SEED)
ALL_FJORD = (LOWER_FJORD, TERMINUS, UPPER_FJORD, UPPER_FJORD_SEED)

def classify_fjord(fjord, grid_info, upstream_loc, terminus):
    """Splits a fjord along a terminus, into an upper and lower section.
    The upper portion does NOT include the (rasterized) terminus line

    fjord: np.array(bool)
        Definition of a fjord on a local grid.
        Eg: result of bedmachine.get_fjord()

    grid_info: gdalutil.FileInfo
        Definition of the grid used for fjord
        Eg: gdalutil.FileInfo(grid_file)

    upstream_loc: shapely.geometry.Point
        A single point in the upstream portion of the fjord

    terminus: shapely.geometry.LineString
        The terminus on which to split

    Returns: np.array(int)
        0 = Unused
        1 = lower fjord
        2 = glacier terminus (in fjord)
        3 = glacier terminus (out of fjord)
        4 = upper fjord
        5 = the fill seed point (in the upper fjord)

    """

    # Extend and rasterize the terminus; can be used to cut fjord
    terminus_extended=cgutil.extend_linestring(terminus, 100000.)
    terminus_xr = gdalutil.rasterize_polygons(
        shapelyutil.to_datasource(terminus_extended), grid_info)

    # Cut the fjord with the terminus
    fj = np.zeros(fjord.shape)
    fj[fjord] = LOWER_FJORD
    fj[terminus_xr != 0] = TERMINUS_EXTRA
    fj[np.logical_and(terminus_xr != 0, fjord)] = TERMINUS

    # Position of upstream point on the raster
    seed = grid_info.to_ij(upstream_loc.x, upstream_loc.y)

    # Don't fill through diagonals
    selem = np.array([
        [0,1,0],
        [1,1,1],
        [0,1,0]
    ])

    fj = skimage.segmentation.flood_fill(
        fj, (seed[1],seed[0]), UPPER_FJORD, selem=selem)
    fj[seed[1],seed[0]] = UPPER_FJORD_SEED
    return fj

def up_fjord(grid_info, fjord, up_loc, terminus):
    """Determines a raster of areas at or above a terminus line

	grid_info: gdalutil.FileInfo
	fjord: np.array(j,i, dtype=bool)
		Raster of 
    """

    fjc = classify_fjord(fjord, grid_info, up_loc, terminus)
    up_fjord = np.isin(fjc, GE_TERMINUS)
    return up_fjord

def ice_area(grid_info, fjord, up_loc, terminus):
    """Determines total area of ice at or above a terminus line"""

    return grid_info.dx * grid_info.dy * np.sum(up_fjord(grid_info, fjord, up_loc, terminus))

# -----------------------------------------------------
def von_mises_stress(u_surface, v_surface, thk, dx, dy):
    """Computes the von Mises stress; used for calving law.

    NOTE: Stress is wrong at ice boundaries, due to lack of dealing with
          the issue in finite differences.
    
    u_surface: np.array
        X component of surface velocity
    v_surface: np.array
        Y component of surface velocity
    thk: np.array
        Ice thickness
    dx,dy:
    	Size of gridcell
    """
 
    # Flux, not surface velocity
    uu = u_surface * thk
    vv = v_surface * thk

    # Compute Jacobian (gradiant) of Velocity vector
    d_dy = findiff.FinDiff(0, dy)
    d_dx = findiff.FinDiff(1, dx)
    dv_dy = d_dy(vv)
    dv_dx = d_dx(vv)
    du_dy = d_dy(uu)
    du_dx = d_dx(uu)
#    print('shapes1: {} {}'.format(vv.shape, dv_dx.shape))

    # div = dv_dy + du_dx

    # Compute strain rate tensor
    # Follows: https://en.wikipedia.org/wiki/Strain-rate_tensor
    # L = [[dv_dy, du_dy],
    #      [dv_dx, du_dx]]
    # E = 1/2 (L + L^T)    [symmetric tensor]
    E_yy = dv_dy
    E_xx = du_dx
    E_yx = .5 * (du_dy + dv_dx)


    # Obtain eigenvalues (L1,L2) of the strain rate tensor
    # Closed form eigenvalues of a 2x2 matrix
    # http://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
    T = E_yy + E_xx    # Trace of strain rate tensor = divergence of flow
    flux_divergence = T    # Trace of tensor is same as flux divergence of flow (v,u)
    D = E_yy*E_xx - E_yx*E_yx   # Determinate of strain rate tensor
    qf_A = .5*T
    qf_B = np.sqrt(.25*T*T - D)
    L1 = qf_A + qf_B   # Biggest eigenvalue
    L2 = qf_A - qf_B   # Smaller eigenvalue


    return von_mises_stress_eig(L1, L2)

def von_mises_stress_eig(L1, L2):
    """Calculate the von Mises stress,given the eigenvalues L1 and L2.

    * L1 and L2 are computed by PISM as strain_rates[0] and
      strain_rates[1] (see calving0.py).
    * They may also be computed, less well, by von_mises_stress() (above).
    """

    # Follows Morlighem et al 2016: Modeling of Store
    # Gletscher's calving dynamics, West Greenland, in
    # response to ocean thermal forcing
    # https://doi.org/10.1002/2016GL067695
#    print(type(L1))
#    print('shape2', L1.shape)
    maxL1 = np.maximum(0.,L1)
    maxL2 = np.maximum(0.,L2)
    # e2 = [effective_tensile_strain_rate]^2
    e2 = .5 * (maxL1*maxL1 + maxL2*maxL2)    # Eq 6

    glen_exponent = 3    # n=3

    # PISM computes ice hardness (B in Morlighem et al) as follows:
    # https://github.com/pism/pism/blob/44db29423af6bdab2b5c990d08793010b2476cc5/src/rheology/IsothermalGlen.cc
    # https://github.com/pism/pism/blob/44db29423af6bdab2b5c990d08793010b2476cc5/src/rheology/FlowLaw.cc
    hardness_power = -1. / glen_exponent


    # Table from p. 75 of:
    # Cuffey, K., and W. S. B. Paterson (2010), The
    # Physics of Glaciers, Elsevier, 4th ed., Elsevier,
    # Oxford, U. K.
    # Table 3.4: Recommended base values of creep
    #            parameter $A$ at different temperatures
    #            and $n=3$
    # T (degC) | A (s-1Pa-3)
    #  0    2.4e-24
    #- 2    1.7e-24
    #- 5    9.3e-25
    #-10    3.5e-25
    #-15    2.1e-25
    #-20    1.2e-25
    #-25    6.8e-26
    #-30    3.7e-26
    #-35    2.0e-26
    #-40    1.0e-26
    #-45    5.2e-27
    #-50    2.6e-27

    # https://github.com/pism/pism/blob/5e1debde2dcc69dfb966e8dec7a58963f1967caf/src/pism_config.cdl
    # pism_config:flow_law.isothermal_Glen.ice_softness = 3.1689e-24;
    # pism_config:flow_law.isothermal_Glen.ice_softness_doc = "ice softness used by IsothermalGlenIce :cite:`EISMINT96`";
    # pism_config:flow_law.isothermal_Glen.ice_softness_type = "number";
    # pism_config:flow_law.isothermal_Glen.ice_softness_units = "Pascal-3 second-1";
    softness_A = 3.1689e-24
    hardness_B = pow(softness_A, hardness_power)

    # Compute tensile von Mises stress, used for threshold calving
    tensile_von_Mises_stress = np.sqrt(3) * hardness_B * \
        np.power(e2, (1./(2*glen_exponent)))    # Eq 7

    return tensile_von_Mises_stress
