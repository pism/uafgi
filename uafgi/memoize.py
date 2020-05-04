import functools
import collections
from giss.functional import *
from giss import checksum,giutil
import os
import pickle

"""Generalized functional-style access to data."""

# https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize


class File(object):
    def __init__(self, realpath):
        """
        realpath:
            The canonical path to the file."""
        self.realpath = realpath
        self.mtime = os.path.getmtime(self.realpath)
        self.size = os.path.getsize(self.realpath)

    hash_version = 0
    def hashup(self, hash):
        """Hashes a file IN ITS CURRENT STATE"""
        checksum.hashup_str(hash, str(self.realpath))
        checksum.hashup(hash, self.mtime)
        checksum.hashup(hash, self.size)

    def __repr__(self):
        return 'File({}, {}, {})'.format(self.realpath, self.mtime, self.size)

# Tuple describing the origin of a file and its present state
OriginInfo = collections.namedtuple('OriginInfo',
    ('ofile', 'origin_file', 'file_mtime', 'file_size', 'origin_hash'))

@giutil.arg_decorator
class files(object):
    """Decorator to memoize on the files in a special "File function."

    A filefn is run on filefn(*args, **kwargs) to produce a thunk with
    the additional properties:
      * hash_version  = For the whole class...
      * thunk.inputs  = File inputs for this function call
      * thunk.outputs = File outputs for this function call
                        [(ofile, origin_tuple)]
      * thunk()       = Execute the function
    """
    def __init__(self, filefn):
        self.filefn = filefn

    def _origin_info(self, inputs_hash, ofile, origin_tuple):
        """returns: origin_file, origin_hash"""

        # Name of origin file
        odir,oleaf = os.path.split(ofile)
        ofile_origin = os.path.join(odir, '.' + oleaf + '.origin')

        # ---------------------------------------------------
        # Hash that goes in the origin file
        hash = checksum.begin_checksum()

        # ---- The function used to compute this output
        hash.update(self.filefn.__module__.encode())
        hash.update(self.filefn.__qualname__.encode())
        checksum.hashup_int(hash, self.filefn.hash_version)

        # ---- Input files used to compute this output
        hash.update(inputs_hash)

        # ---- Details of the output itself
        checksum.hashup(hash, origin_tuple)
        # ---------------------------------------------------

        mtime = None
        try:
            mtime = os.path.getmtime(ofile)
        except IOError:
            pass

        size = None
        try:
            size = os.path.getsize(ofile)
        except IOError:
            pass

        return OriginInfo(ofile, ofile_origin, mtime,
            size, hash.digest())


    def __call__(self, *args, **kwargs):
        all_good = True

        # See what files this function would read/write.
        thunk = self.filefn(*args, **kwargs)


        # ---------- See if any of OUR output files have mysteriously changed
        # (this "shouldn't" happen...)
        # TODO: What do to if inputs are gone?  Depending on an option, 
        #       either: (a) throw exception, (b) just return output file.
        #
        # Identify what the OriginInfo for each file should look like
        inputs_hash = checksum.checksum([File(x) for x in sorted(thunk.inputs)])
        origins = [self._origin_info(inputs_hash, ofile, origin_tuple)
            for ofile,origin_tuple in thunk.outputs]

        # Validate the origin info against output of what's on disk
        if all_good:
            for origin in origins:
                if not os.path.exists(origin.ofile):
                    all_good = False
                    break

                try:
                    with open(origin.origin_file, 'rb') as fin:
                        file_mtime, file_size, origin_hash = pickle.load(fin)

                    if file_mtime != origin.file_mtime or \
                        file_size != origin.file_size or \
                        origin_hash != origin.origin_hash:

                        all_good = False
                        break

                except Exception as e:
                    # If any problems... re-do it!
                    all_good = False
                    break

        if all_good:
            # Reconstruct return value from past run
            return thunk.value
        else:
            value = thunk()    # Return value comes as Thunk property set at __init__()

            # Now write origin file for each output file we created
            for origin in origins:
                with open(origin.origin_file, 'wb') as out:
                    file_size = os.path.getsize(origin.ofile)
                    mtime = os.path.getmtime(origin.ofile)
                    pickle.dump((mtime, file_size, origin.origin_hash), out)

            return value

@giutil.arg_decorator
class local(object):
    """Decorator. Caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned
    (not reevaluated).

    NOTE: This needs to be a class to keep it checksummable"""

    def __init__(self, func, cache=dict(), id_fn=checksum.checksum):
        self.func = func
        self.cache = cache
        self.id_fn = id_fn

    def __call__(self, *args, **kwargs):
        # Look up the dict used for memoizing this function
        func_id = self.id_fn(self.func)

        # Get checksum of  complete function call
        thunk = bind(self.func, *args, **kwargs)
        thunk_id = self.id_fn(thunk)

        if thunk_id not in self.cache:
            value = thunk()
            self.cache[thunk_id] = value

        return self.cache[thunk_id]

    # Allow to decorate methods as well as functions
    # See: http://www.ianbicking.org/blog/2008/10/decorators-and-descriptors.html
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        new_func = self.func.__get__(obj, type)
        return self.__class__(new_func)

    # Enable checksums on this class
    hash_version=1
    def hashup(self, hash):
        checksum.hashup(hash, self.func)

