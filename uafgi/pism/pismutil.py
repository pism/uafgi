import numpy as np
import os
import subprocess
import netCDF4
from uafgi import ncutil


# See PISM source src/util/Mask.hh
MASK_UNKNOWN          = -1,
MASK_ICE_FREE_BEDROCK = 0,
MASK_GROUNDED         = 2,
MASK_FLOATING         = 3,
MASK_ICE_FREE_OCEAN   = 4

MULTIMASK_ICE = (MASK_GROUNDED, MASK_FLOATING)


def fix_output(output_file3, exception, time_units_s, output_file4, delete_vars=list()):
    """Compress PISM output file while correcting time units; and removes original.
    exception:
        Exception (if any) that occurred while running PISM
    units_s: cfunits.Unit
        Units in "seconds since..." format

    NOTE: Fixes to the time unit could be done when the PISM file is opened.
          https://github.com/pism/pism/blob/master/site-packages/PISM/util.py#L30

          To avoid "fixing" the file you could just copy-and-paste the
          guts of prepare_output() and supply your own calendar and
          units string when the file's metadata is written the first
          time.

    """

    # Convert to NetCDF4
    print('Compressing PISM output to NetCDF4...')
    with netCDF4.Dataset(output_file3, 'r') as ncin:
        schema = ncutil.Schema(ncin)

        # Remove excess variables
        print('delete_vars ', delete_vars)
        for vname in delete_vars:
            del schema.vars[vname]

        # Fix time units
        ncv = schema.vars['time']
        ncv.attrs['units'] = str(time_units_s) # 'seconds since {:04d}-{:02d}-{:02d}'.format(dt0.year,dt0.month,dt0.day)
        ncv.attrs['calendar'] = 'proleptic_gregorian'

        schema.write(ncin, output_file4)

def flux_across_terminus(mask, fjord, uvel, vvel, dx, dy):
    """
    Computes the flux of a vector field across the glacier terminus.

    mask:
        PISM mask
    fjord:
        True where the fjord is
    uvel, vvel: [m s-1]
        x and y components of flux
    dx, dy: [m]

    Returns: fluxmap [m^2 s-1]
        Raster of flux through each boundary cell.
        flux = np.sum(np.sum(fluxmap))
    """

    # Areas we lack data over the terminus
#    x = np.logical_and(uvel == 0., uvel==0.).astype(flow)

    # Flux to the East: [m^2 s-1] ("horizontal sheet extruding through terminus")
    maskX = mask.copy()
    maskX[:,:-1] = mask[:,1:]    # Shift west by 1 pixel
    fjordX = fjord.copy()
    fjordX[:,:-1] = fjord[:,1:]    # Shift west by 1 pixel
    cells = np.logical_and.reduce(
        (fjord, fjordX, np.isin(mask,MULTIMASK_ICE), maskX==MASK_ICE_FREE_OCEAN))
    ncellsE = np.sum(cells)
    #                  [m s-1]    [0/1]            [m]
    fluxE = np.maximum(uvel * cells.astype(float) * dy, 0.0)
#    nzE =  np.maximum(uvel * cells.astype(float) * dy, 0.0)

    # Flux to the West
    maskX = mask.copy()
    maskX[:,1:] = mask[:,:-1]    # Shift east by 1 pixel
    fjordX = fjord.copy()
    fjordX[:,1:] = fjord[:,:-1]    # Shift east by 1 pixel
    cells = np.logical_and.reduce(
        (fjord, fjordX, np.isin(mask,MULTIMASK_ICE), maskX==MASK_ICE_FREE_OCEAN))
    ncellsW = np.sum(cells)
    fluxW = np.maximum(-uvel * cells.astype(float), 0.)

    # Flux to the North
    maskX = mask.copy()
    maskX[:-1,:] = mask[1:,:]    # Shift south by 1 pixel
    fjordX = fjord.copy()
    fjordX[:-1,:] = fjord[1:,:]    # Shift south by 1 pixel
    cells = np.logical_and.reduce(
        (fjord, fjordX, np.isin(mask,MULTIMASK_ICE), maskX==MASK_ICE_FREE_OCEAN))
    ncellsN = np.sum(cells)
    fluxN = np.maximum(vvel * cells.astype(float) * dx, 0.0)

    # Flux to the South
    maskX = mask.copy()
    maskX[1:,:] = mask[:-1,:]    # Shift north by 1 pixel
    fjordX = fjord.copy()
    fjordX[1:,:] = fjord[:-1,:]    # Shift north by 1 pixel
    cells = np.logical_and.reduce(
        (fjord, fjordX, np.isin(mask,MULTIMASK_ICE), maskX==MASK_ICE_FREE_OCEAN))
    ncellsS = np.sum(cells)
    fluxS = np.maximum(-vvel * cells.astype(float) * dx, 0.0)

    return \
        fluxE + fluxW + fluxN + fluxS, \
        ncellsE + ncellsW + ncellsN + ncellsS    # Number of segments over which flux is integrated
