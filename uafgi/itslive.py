from uafgi import regrid,cdoutil
from cdo import Cdo

# General rule to tranform a single file
class single_file_transform(object):
    def __init__(makefile, ifname, ofname, action_fn):
        self.action_fn = action_fn
        self.rule = makefile.add(self.run, (input,), (output,)

    def run(self):
        action_fn(self.rule.inputs[0], self.rule.outputs[0])



def merge_to_pism(makefile, grid, ifpattern, years, odir):
    """Macro:

    ifpattern:
        Eg, 'data/itslive/GRE_G0240_{}.nc'
    years: iterable(int)
        Years to process
    """

    # Rule to create each year's file
    years = list(years)
    year_files = list()
    for year in years:
        syear = '{:04}'.format(year)
        ifname = ifpattern.format(syear)
        rule = regrid.extract_region(makefile, grid, ifname,
            'outputs/{}-grid.nc'.format(grid),
            ('vx', 'vy', 'v'), 'outputs').rule

        year_files.append(rule.outputs[0])

    # Rule to merge by time
    cdo = Cdo()
    allyear_file = ifpattern.format('{:04}_{:04}'.format(years[0],years[-1]))
    xrule = cdoutil.merge(makefile, cdo.mergetime,
        year_files, allyear_file,
        options='-f nc4 -z zip_2')

    return xrule

