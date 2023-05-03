import os,sys,time,datetime
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
    t0_s = datetime.datetime.fromtimestamp(t0).astimezone().isoformat()
    print(f'START TIME: {t0} ({t0_s})')
    rule.action(tdir)
    t1 = time.time()
    t1_s = datetime.datetime.fromtimestamp(t1).astimezone().isoformat()
    print(f'END TIME: {t1} ({t1_s})')
    print('ELAPSED TIME (s): {}'.format(t1-t0))
