import os,subprocess

"""Common rules for Makefiles"""

class nccompress(object):
    """Compresses NetCDF file(s)"""
    def __init__(self, makefile, ipaths):
        opaths = list()
        for ipath in ipaths:
            root,ext = os.path.splitext(ipath)
            opath = root + '.nc4'
            opaths.append(opath)
        self.rule = makefile.add(self.run,
            ipaths, opaths)

    def run(self):
        for ipath,opath in zip(self.rule.inputs, self.rule.outputs):
            cmd = ['ncks', '-4', '-L', '1', ipath, opath]
            subprocess.run(cmd, check=True)

