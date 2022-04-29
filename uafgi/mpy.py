import sys
import os
import importlib

def main():
    """Runs a main() program in another Python file (as with the Python command);
    but does so putting the main program in a definite module.

    NOTES:
     1. The directory of the Python file being run must be findable
        in PYTHONPATH.

     2. This is to be launched from an `mpy` shellscript that
        preserves DYLD_FALLBACK_LIBRARY_PATH (for shapely) on macOS.  See
        pismip6-catalina/harness-loads-x.
    """

    # Name of python file to execute
    pyfile0 = sys.argv[1]
    pyfile = os.path.realpath(pyfile0)


    # Find a path that's a prefix of the file being run
    for path in sys.path:
        if len(path) == 0:
            continue
        rpath = os.path.realpath(path)

        if pyfile.startswith(rpath):
            # Load main program as module
            module_name = os.path.splitext(pyfile[len(rpath)+1:])[0].replace(os.sep, '.')
            #print(rpath)
            #print('mpy.py: Running {}'.format(module_name))
            module = importlib.import_module(module_name)

            # Shift arguments so mpy is tranparent to program being run.
            sys.argv = sys.argv[1:]

            # Run the main program!
            module.main()

            return

    # File we're trying to run is not prefixed by PYTHONPATH
    raise ValueError('Trying to run {} but it does not have a prefix in PYTHONPATH'.format(pyfile))

main()
