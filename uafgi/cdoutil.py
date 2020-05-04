import os.path
from giss import ioutil

"""Utilities for working with the Python CDO interface"""
def _large_merge(cdo_merge_operator, input, output, tmp_files, max_merge=30, **kwargs):
    """
    max_merge:
        Maximum number of files to merge in a single CDO command
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
        cdo_merge_operator(input=input, output=output, **kwargs)


def large_merge(cdo_merge_operator, input, output, max_merge=30, **kwargs):
    """Recursively merge large numbers of files using a CDO merge-type operator
    cdo_merge_operator:
        The CDO operator used to merge; Eg: cdo.mergetime
    input:
        Names of the input files to merge
    output:
        The output files to merge
    kwargs:
        Additional arguments to supply to the CDO merge command
    max_merge:
        Maximum number of files to merge in a single CDO command.
        This cannot be too large, lest it overflow the number of available OS filehandles.
    """

    print('Merging {} files into {}'.format(len(input), output))
    odir = os.path.split(output)[0]
    with ioutil.TmpFiles(os.path.join(odir, 'tmp')) as tmp_files:
        _large_merge(cdo_merge_operator, input, output, tmp_files, max_merge=max_merge, **kwargs)
