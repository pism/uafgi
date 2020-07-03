import os,subprocess
from cdo import Cdo
from uafgi import nsidc
from uafgi import giutil,cdoutil,make

# -------------------------------------------------------------------------
class merge_component(object):
    """Makefile macro, create one component of a glacier file by merging individual frames.

    idir:
        Input directory of GeoTIFF files
    odir:
        Directory for output files
    filter_attrs:
        File name portion (PFile) attributes used to filter files in idir.
        Should contain at least source, grid, parameter
    ofpattern:
        Eg: '{source}_{grid}_2008_2020.nc'
    blacklist:
        Files to NOT include
    max_files:
        Maximum number of files to include
    """

    def __init__(self, makefile, idir, parse_fn, odir, filter_attrs=dict(), **kwargs):

        rule = nsidc.tiffs_to_netcdfs(makefile, idir, parse_fn, odir, filter_attrs=filter_attrs, **kwargs).rule

        # Start a new mergefile
        inputs = rule.outputs
        output = os.path.join(odir, '{source}_{grid}_{parameter}_merged.nc'.format(**filter_attrs))

        self.rule = makefile.add(self.run, inputs, (output,))


    def run(self):
        cdo = Cdo()

        # Merge into the mergefile
        cdoutil.large_merge(
            cdo.mergetime,
            input=self.rule.inputs,
            output=self.rule.outputs[0],
            options="-f nc4 -z zip_2",
            max_merge=50)
# -------------------------------------------------------------------------
class merge(object):
    """Makefile macro, create a final glacer file, with all components.

    idir:
        Input directory of GeoTIFF files
    odir:
        Directory for output files
    filter_attrs:
        File name portion (PFile) attributes used to filter files in idir.
        Should contain at least source, grid
    parameters:
        List of components ("parameter" in filter_attrs) to process.
    ofpattern:
        Format pattern used to determine output filename, including directory.
        Keys should match up with filter_attrs
        Eg: 'outputs/{source}_{grid}_2008_2020.nc'
    blacklist:
        Files to NOT include
    max_files:
        Maximum number of files to include
    """
    def __init__(self, makefile, idir, parse_fn, odir, ofpattern, parameters, filter_attrs=dict(), **kwargs):
        inputs = list()
        for parameter in parameters:
            rule = merge_component(makefile, idir, parse_fn, odir,
                filter_attrs=giutil.merge_dicts(filter_attrs, {'parameter': parameter}), **kwargs).rule
            inputs += rule.outputs

        self.rule = makefile.add(self.run, inputs, (ofpattern.format(**filter_attrs),))

    def run(self):
        print('Merging to {}'.format(self.rule.outputs[0]))
        cdo = Cdo()
        cdo.merge(
            input=self.rule.inputs,
            output=self.rule.outputs[0],
            options="-f nc4 -z zip_2")
# -------------------------------------------------------------------------
class fixup_velocities_for_pism(object):
    """Fixup merged velocity file
    ipath:
        Name of merged velocity file
    """
    def __init__(self, makefile, ipath, odir):
        self.rule = makefile.add(self.run,
            (ipath,), (make.opath(ipath, odir, '_pism'),))

    def run(self):
        # Rename variables
        # HINT: If presence is intended to be optional, then prefix
        # old variable name with the period character '.', i.e.,
        # 'ncrename -v .vy,v_ssa_bc'. With this syntax ncrename would
        # succeed even when no such variable is in the file.
        cmd = ['ncrename', '-O', '-v', '.vx,u_ssa_bc', '-v', '.vy,v_ssa_bc',
            self.rule.inputs[0], self.rule.outputs[0]]
        subprocess.run(cmd, check=True)
