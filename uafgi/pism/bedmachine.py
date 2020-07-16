import os,subprocess
import netCDF4
from uafgi import make
from uafgi.make import ncmake

class extract(object):

    """Extracts a local version of bedmachine file

    global_bedmachine_path:
        Name of basic bedmachine file (after fixup_pism)
    grid: str
        NSIDC name of local glacier grid (eg: 'W69.10N')
    data_path:
        Any old file with x(x) and y(y) coordinate variables
    """
    def __init__(self, makefile, grid, global_bedmachine_path, data_path, odir):

        ofname = make.opath(global_bedmachine_path, odir, '_'+grid)
        ofname = os.path.splitext(ofname)[0] + '.nc'
        self.rule = makefile.add(self.run,
            (global_bedmachine_path, data_path),
            (ofname,))

    def run(self):

        data_path = self.rule.inputs[1]
        with netCDF4.Dataset(data_path) as nc:
            xx = nc.variables['x'][:]
            dx = xx[1]-xx[0]
            half_dx = .5 * dx
            x0 = round(xx[0] - half_dx)
            x1 = round(xx[-1] + half_dx)

            yy = nc.variables['y'][:]
            dy = yy[1]-yy[0]
            half_dy = .5 * dy
            y0 = round(yy[0] - half_dy)
            y1 = round(yy[-1] + half_dy)

        # For Jakobshavn: gdal_translate
        #    -r average -projwin -219850 -2243450 -132050 -2318350 
        #    -tr 100 100
        #    NETCDF:outputs/bedmachine/BedMachineGreenland-2017-09-20_pism.nc4:thickness
        #    ./out4.nc

        self.cmd = ['gdal_translate',
            '-r', 'average',
            '-projwin', str(x0), str(y1), str(x1), str(y0),
            '-tr', str(dx), str(dy),
            'NETCDF:' + self.rule.inputs[0] + ':thickness',
            self.rule.outputs[0]]


        print('*********** ', ' '.join(self.cmd))
        subprocess.run(self.cmd, check=True)




class fixup_pism0(object):
    """Fixup bedmachine file for use as PISM input file
    ipath:
        Name of BedMachine file
    odir:
        Place to put output BedMachine file
    """
    def __init__(self, makefile, ipath, odir):
        self.rule = makefile.add(self.run,
            (ipath,), (make.opath(ipath, odir, '_pism'),))

    def run(self):
        # Reverse the direction of the Y coordinate
        cmd = ['ncpdq', '-O', '-a', '-y', self.rule.inputs[0], self.rule.outputs[0]]
        subprocess.run(cmd, check=True)

def fixup_pism(makefile, ipath, odir):
    """
    ipath:
        Name of BedMachine file
    odir:
        Place to put output BedMachine file
    """
    rule = fixup_pism0(makefile, ipath, odir).rule
    rule = ncmake.nccompress(makefile, rule.outputs).rule
    return rule
