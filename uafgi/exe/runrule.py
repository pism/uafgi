import os,sys,time
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

    # run with timing
    t0 = time.time()
    print('START TIME: {}'.format(t0))
    rule.action(tdir)
    t1 = time.time()
    print('END TIME: {}'.format(t1))
    print('ELAPSED TIME (s): {}'.format(t1-t0))
