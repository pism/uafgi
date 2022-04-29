import sys
import os
import importlib
import dill

# Load the rule from the file
with open(sys.argv[1], 'rb') as fin:
    tdir_fn = dill.load(fin)
    rule = dill.load(fin)

# Execute the rule
with tdir_fn() as tdir:    # See ioutil.TmpDir
    print('=============================================================')
    print('Command line: {}'.format(sys.argv))
    print('Running {}'.format(str(rule)))
    sys.stdout.flush()
    rule.action(tdir)
