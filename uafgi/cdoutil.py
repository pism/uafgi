import os.path
from uafgi import ioutil

"""Utilities for working with the Python CDO interface"""
def _large_merge(cdo_merge_operator, input, output, tmp_files, max_merge=30, **kwargs):
    """
    max_merge:
        Maximum number of files to merge in a single CDO command
    kwargs:
        Additional arguments given to the CDO operator.
        Eg for cdo.merge(): options='-f nc4'
        https://code.mpimet.mpg.de/projects/cdo/wiki/Cdo%7Brbpy%7D#onlineoffline-help
    """
    print('_large_merge', len(input), output)
    odir = os.path.split(output)[0]

    if len(input) > max_merge:

        input1 = list()
        try:
            chunks = [input[x:x+max_merge] for x in range(0, len(input), max_merge)]
            for chunk in chunks:
                ochunk = next(tmp_files)
                input1.append(ochunk)
                _large_merge(cdo_merge_operator, chunk, ochunk, tmp_files, max_merge, **kwargs)

            _large_merge(cdo_merge_operator, input1, output, tmp_files, max_merge, **kwargs)
        finally:
            # Remove our temporary files
            for path in input1:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
    else:
#        print('CDO Merge')
#        print('INPUT ',input)
#        print('OUTPUT ',output)
        cdo_merge_operator(input=input, output=output, **kwargs)


def do_merge(cdo_merge_operator, inputs, output, max_merge=30, **kwargs):
    """Recursively merge large numbers of files using a CDO merge-type operator.
    Also appropriate for "small" merges.

    cdo_merge_operator:
        The CDO operator used to merge; Eg: cdo.mergetime
    inputs:
        Names of the input files to merge
    output:
        The output files to merge
    kwargs:
        Additional arguments to supply to the CDO merge command
    max_merge:
        Maximum number of files to merge in a single CDO command.
        This cannot be too large, lest it overflow the number of available OS filehandles.
    """

    print('Merging {} files into {}'.format(len(inputs), output))
    odir = os.path.split(output)[0]
    with ioutil.TmpFiles(os.path.join(odir, 'tmp')) as tmp_files:
        _large_merge(cdo_merge_operator, inputs, output, tmp_files, max_merge=max_merge, **kwargs)

class merge(object):
    """Rule for do_merge"""

    def __init__(self, makefile, cdo_merge_operator, inputs, output, max_merge=30, **kwargs):
        self.inputs = inputs
        self.output = output
        self.max_merge = max_merge
        self.kwargs = kwargs
        self.cdo_merge_operator = cdo_merge_operator
        self.rule = makefile.add(self.run,
            inputs, (output,))

    def run(self):
        do_merge(self.cdo_merge_operator, self.inputs, self.output, max_merge=self.max_merge, **self.kwargs)

# -------------------------------------------------------------

class set_time_axis(object):
    """Rule to set the time axis for a single file."""

    def __init__(self, makefile, input, output, time_bounds, reftime):
        """time_boudns (dt0, dt1): [datetime.date]
            Time bounds for this file
        """

        # This deduces the mid-point (nominal) date from the filename
        self.time_bounds = time_bounds
        self.nominal_date = time_bounds[0] + (time_bounds[1] - timebounds[0]) / 2
        self.reftime = reftime
        self.rule = makefile.add(self.run, (input,), (output,))

    def run(self):
        cdo = Cdo()

        # Set the time axis
        inputs = [
            '-setreftime,{}'.format(self.reftime),  # Somehow reftime is being ignored
            '{}'.format(self.ipath)]
        cdo.settaxis(
            self.nominal_date.isoformat(),
            input=' '.join(inputs),
            output=self.opath,
            options="-f nc4 -z zip_2")

        # Add time bounds --- to be picked up by cdo.mergetime
        # https://code.mpimet.mpg.de/boards/2/topics/1115
        with netCDF4.Dataset(self.rule.outputs[0], 'a') as nc:
            nctime = nc.variables['time']
            timeattrs = [(name,nctime.getncattr(name)) for name in nctime.ncattrs()]
            nc.variables['time'].bounds = 'time_bnds'

            nc.createDimension('bnds', 2)
            tbv = nc.createVariable('time_bnds', 'd', ('time', 'bnds',))
            # These attrs don't end up in the final merged time_bnds
            # But they do seem to be important to keep the final value correct.
            for name,val in timeattrs:
                tbv.setncattr(name, val)

            cfu = cf_units.Unit(nctime.units, nctime.calendar)
            for ix,time in enumerate(self.time_bounds):
                tbv[0,ix] = cfu.date2num(time)

