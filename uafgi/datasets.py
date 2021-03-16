import os.path

# Root directories
DATA = 'velocities_data'
OUTPUTS = 'outputs'

# Core files
BEDMACHINE_ORIG = os.path.join(DATA, 'bedmachine/BedMachineGreenland-2017-09-20.nc')
BEDMACHINE_PISM = os.path.join(OUTPUTS, 'bedmachine/BedMachineGreenland-2017-09-20_pism.nc')

def measures_grid_file(ns481_grid):
    """File describing an NSIDC-0481 (MEASURES) grid."""
    return os.path.join(DATA, 'measures/grids/{}_grid.nc'.format(ns481_grid))

def bedmachine_local(grid):
    """BedMachine file localized for a NSIDC 0481 (MEASURES) grid"""
    return os.path.join(
        OUTPUTS, 'bedmachine/BedMachineGreenland-2017-09-20_{}.nc'.format(grid))
