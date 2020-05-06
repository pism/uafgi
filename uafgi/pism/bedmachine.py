import os,subprocess
from uafgi import make
from uafgi.make import ncmake

class fixup_pism0(object):
    """Fixup bedmachine file for use as PISM input file
    ipath:
        Name of BedMachine file
    """
    def __init__(self, makefile, ipath, odir):
        self.rule = makefile.add(self.run,
            (ipath,), (make.opath(ipath, odir, '_pism'),))

    def run(self):
        # Reverse the direction of the Y coordinate
        cmd = ['ncpdq', '-O', '-a', '-y', self.rule.inputs[0], self.rule.outputs[0]]
        subprocess.run(cmd, check=True)

def fixup_pism(makefile, *args, **kwargs):
    rule = fixup_pism0(makefile, *args, **kwargs).rule
    rule = ncmake.nccompress(makefile, rule.outputs).rule
    return rule
