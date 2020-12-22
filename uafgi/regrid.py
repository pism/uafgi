import os,subprocess
import netCDF4
from uafgi import make
from uafgi.make import ncmake
import re

class extract_region_separate_files(object):
    """Extracts a local version of a large-area / hi-res file
    Writes output in 1 variable per file

    grid: str
        Name of local grid (eg: 'W69.10N')
    global_data:
        Name of basic bedmachine file (after fixup_pism)
    data_path:
        Any old file with x(x) and y(y) coordinate variables of target region
    vnames:
        Names of variables in the file to extract (each will go to its own file)
    odir:
        Output director where to place results
    """
    def __init__(self, makefile, grid, global_data, data_path, vnames, odir):
        self.vnames = vnames
        outputs = list()
        for vname in self.vnames:
            ofname = make.opath(global_data, odir, '_'+grid+'_'+vname)
            ofname = os.path.splitext(ofname)[0] + '.nc'
            outputs.append(ofname)

        self.rule = makefile.add(self.run,
            (global_data, data_path),
            outputs)

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

        for ix,vname in enumerate(self.vnames):
            self.cmd = ['gdal_translate',
                '-r', 'average',
                '-projwin', str(x0), str(y1), str(x1), str(y0),
                '-tr', str(dx), str(dy),
                'NETCDF:' + self.rule.inputs[0] + ':'+vname,
                self.rule.outputs[0+ix]]


            print('*********** ', ' '.join(self.cmd))
            subprocess.run(self.cmd, check=True)


trimRE = re.compile(r'(.*)_([^_]*)(\..*)')
class merge(object):
    """Merges variables of two BedMachine files produced by extract(), into one."""
    def __init__(self, makefile, inputs, odir):
        ofname = make.opath(inputs[0], odir, '')
        match = trimRE.match(ofname)
        ofname = match.group(1) + match.group(3)
        self.rule = makefile.add(self.run,
            inputs, (ofname,))

    def run(self):
        # -O flag forces overwrite
        cmd = ['cdo', '-O', 'merge'] + list(self.rule.inputs) + list(self.rule.outputs)
        subprocess.run(cmd, check=True)
        

def extract_region(makefile, grid, global_data, data_path, vnames, odir):
    """Macro: extracts all variables from one file, stores in another."""

    # Create one extract file per variable
    rule = extract_region_separate_files(makefile, grid, global_data, data_path, vnames, odir).rule

    # Merge multiple extract files into one
    xrule = merge(makefile, rule.outputs, odir)
    return xrule
