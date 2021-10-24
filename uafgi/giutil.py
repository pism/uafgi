# PyGISS: Misc. Python library
# Copyright (c) 2013-2016 by Elizabeth Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import os
import re
import string
import glob
import contextlib
import sys
import collections.abc
import types
import string

class Struct(object):
    """Convert a dict() to a struct."""
    def __init__(self, entries): 
        self.__dict__.update(entries)

class curry(object):
    """Curry a function, i.e. produce a new function in which the
    first n parameters have been set.

    Args:
        fun:
            The function to curry
        *args:
            The first few arguments to the function
        **kwargs:
            Any keyword arguments to pass to the function

    Returns:
        Object that acts like the new curried function.

    Example:
        double = curry(operator.mul, 2)
        print double(17)
        triple = curry(operator.mul, 3)

    See:
        http://code.activestate.com/recipes/52549-curry-associating-parameters-with-a-function"""

    def __init__(self, fun, *args, **kwargs):
        self.fun = fun
        self.pending = args[:]
        self.kwargs = kwargs.copy()
    def __call__(self, *args, **kwargs):
        if kwargs and self.kwargs:
            kw = self.kwargs.copy()
            kw.update(kwargs)
        else:
            kw = kwargs or self.kwargs
        return self.fun(*(self.pending + args), **kw)

# -----------------------------------------------------------
# see: http://stackoverflow.com/questions/8100166/inheriting-methods-docstrings-in-python
def inherit_docs(cls):
    """Class decorator that inerhits docstrings from the superclass."""
    for name in dir(cls):
        func = getattr(cls, name)
        if func.__doc__: 
            continue
        for parent in cls.mro()[1:]:
            if not hasattr(parent, name):
                continue
            doc = getattr(parent, name).__doc__
            if not doc: 
                continue
            try:
                # __doc__'s of properties are read-only.
                # The work-around below wraps the property into a new property.
                if isinstance(func, property):
                    # We don't want to introduce new properties, therefore check
                    # if cls owns it or search where it's coming from.
                    # With that approach (using dir(cls) instead of var(cls))
                    # we also handle the mix-in class case.
                    wrapped = property(func.fget, func.fset, func.fdel, doc)
                    clss = filter(lambda c: name in vars(c).keys() and not getattr(c, name).__doc__, cls.mro())
                    setattr(clss[0], name, wrapped)
                else:
                    try:
                        func = func.__func__ # for instancemethod's
                    except:
                        pass
                    func.__doc__ = doc
            except: # some __doc__'s are not writable
                pass
            break
    return cls

# -----------------------------------------------------------
def numpy_stype(var) :
    """Provides a summary of a numpy variable.

    Returns:
        Textual summary.  Eg: int[4,3]
    """
    return '%s%s' % (str(var.dtype), str(var.shape))
# -----------------------------------------------------------
def search_file(filename, search_path):
    """Given a search path, find file.
    Args:
        Alt 1: search_path[] (string):
            Directories where to search for file
        Alt 2: search_path (string)
            Directories where to search for file, using path separator.
            Eg: '/usr/home:/usr/bin'

    See:
        http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    """
    if isinstance(search_path, str) :
        search_path = string.split(search_path, os.pathsep)
    for path in search_path :
        if os.path.exists(os.path.join(path, filename)):
            return os.path.abspath(os.path.join(path, filename))

    # Not found :(
    return None
# -----------------------------------------------------------
def sum_by_cols(matrix) :
    return np.array(matrix.sum(axis=0)).reshape(-1)

def sum_by_rows(matrix) :
    return np.array(matrix.sum(axis=1)).reshape(-1)

def reshape_no_copy(arr, *shape) :
    """Reshape a np.array, but don't make any copies of it.
    Throws an exception if the new reshaped view cannot be made
    (for example, if the original array were non-contiguous"""
    ret = arr.view()
    ret.shape = shape
    return ret

# -----------------------------------------------------------
def multiglob_iterator(paths) :
    """Iterator list a bunch of files from a bunch of arguments.  Tries to work like ls
    Yields:
        (directory, filename) pairs
    See:
        lsacc.py
    """
    if len(paths) == 0 :
        for fname in os.listdir('.') :
            yield ('', fname)
        return

    for path in paths :
        if os.path.isdir(path) :
            for fname in os.listdir(path) :
                yield (path, fname)

        elif os.path.isfile(path) :
            yield os.path.split(path)

        else :
            for ret in multiglob_iterator(glob.glob(path)) :
                yield ret

# http://shallowsky.com/blog/programming/python-tee.html
class tee(object):
    def __init__(self, _fd1, _fd2) :
        self.fd1 = _fd1
        self.fd2 = _fd2

    def __del__(self) :
        if self.fd1 != sys.stdout and self.fd1 != sys.stderr :
            self.fd1.close()
        if self.fd2 != sys.stdout and self.fd2 != sys.stderr :
            self.fd2.close()

    def write(self, text) :
        self.fd1.write(text)
        self.fd2.write(text)

    def flush(self) :
        self.fd1.flush()
        self.fd2.flush()





class CallCounter(object):
    """Wraps a function, counting how many times it's been called."""
    __slots__ = ('fn', 'count')

    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __call__(self, *args, **kwargs):
        ret = self.fn(*args, **kwargs)
        self.count += 1
        return ret


class SlotStruct(object):
    def __init__(self, *args):
        for attr,val in zip(self.__slots__, args):
            setattr(self, attr, val)

    def __getitem__(self, index):
        # TODO: We shouldn't have to go through getattr() here.
        return getattr(self, self.__slots__[index])

    def __len__(self):
        return len(self.__slots__)

    def __repr__(self):
        ret = [str(type(self)), '(']
        for slot in self.__slots__:
            ret.append(repr(getattr(self, slot)))
            ret.append(', ')
        ret[-1] = ')'
        return ''.join(ret)

class LazyDict(collections.abc.Mapping):
    """A dictionary that stores values that will be later lazily evaluated."""

    class Entry(SlotStruct):
        __slots__ = (
            'lam',      # Expression to generate value
            'val',      # The value computed by lam()
            'isset')    # True if val has been set


    class LazyView(collections.abc.MutableMapping):
        """Sets/returns lambdas instead of values.
        NOTE: This dict cannot store None as a value (see getitem())"""
        def __init__(self, dict):
            self._entries = dict

        def __getitem__(self, key):
            entry = self._entries[key]
            if entry.lam is None:
                return lambda: entry.val
            else:
                return entry.lam

        def __setitem__(self, key, lam):
            if not callable(lam):
                raise ValueError("Items inserted into LazyDict.LazyView must be callable.")
            self._entries[key] = LazyDict.Entry(lam, None, False)

        def __delitem__(self, key):
            del self._entries[key]

        def __iter__(self):
            return iter(self._entries)

        def __len__(self):
            return len(self._entries)


    # -----------------------------------------
    def __init__(self):
        self._entries = dict()
        self.lazy = LazyDict.LazyView(self._entries)

    def __getitem__(self, key):
        entry = self._entries[key]  # Could raise a KeyError
        if not entry.isset:
            ret = entry.lam()
            if ret is not None:
                entry.val = ret    # Allow thunks that just change the dict
            entry.isset = True
        return entry.val

    def __setitem__(self, key, val):
        try:
            entry = self._entries[key]
            entry.isset = True
            entry.val = val
        except KeyError:
            self._entries[key] = LazyDict.Entry(None, val, True)

    def __delitem__(self, key):
        del self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

class LambdaDict(LazyDict):
    """Like LazyDict, but never remember function calls."""
    def __getitem__(self, key):
        entry = self._entries[key]  # Could raise a KeyError
        if entry.isset:
            return entry.val
        return entry.lam()



class Thunk(object):
    """Creates a picklable object with fully bound arguments that
    may be called later.  Additional args and kwargs may be added
    at the time of calling.  Call-time arges are PREPENDED to
    the arg list."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *myargs, **mykwargs):
        args = myargs + self.args[1:]
        kwargs = dict(self.kwargs)
        for k,v in mykwargs: kwargs[k] = v
        return self.args[0](*args, **kwargs)

def pickler_add_trace(pickler, trace_fn):
    """Dynamically adds a tracing function to a pickler, to be called
    on every object pickled."""

    if hasattr(pickler, 'persistent_id'):
        old_persistent_id = pickler.persistent_id
    else:
        old_persistent_id = lambda obj: None
    def new_persistent_id(self, obj):
        trace_fn(obj)
        return old_persistent_id(obj)
    pickler.persistent_id = types.MethodType(new_persistent_id, pickler)

def read_config(fname, config=None, remove_junk=True):
    """Reads a python script as a configuration file."""

    if config is None:
        config = dict()

    with open(fname, 'rb') as fin:
        scode = fin.read()

    exec(compile(scode, fname, 'exec'), config)

    # -------------- Remove stuff caller doesn't want to see
    if remove_junk:
        try:
            del config['__builtins__']
        except:
            pass
        for k,v in config.items():
            if isinstance(v, types.ModuleType):
                del config[k]

    return config

# https://www.peterbe.com/plog/uniqifiers-benchmark
def uniq(seq, idfun=None): 
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result
# ------------------------------------------------
def arg_decorator(decorator_fn):
    """Meta-decorator that makes it easier to write decorators taking args.
    See:
       * old way: http://scottlobdell.me/2015/04/decorators-arguments-python
       * new way: See memoize.files (no lambdas required)"""

    class real_decorator(object):
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs    # dict_fn, id_fn

        def __call__(self, func):
            return decorator_fn(func, *self.args, **self.kwargs)

    return real_decorator
# --------------------------------------------------
# http://stackoverflow.com/questions/11283961/partial-string-formatting
class _FormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

# http://stackoverflow.com/questions/11283961/partial-string-formatting
def partial_format(str, **kwargs):
    formatter = string.Formatter()
    mapping = _FormatDict(kwargs)
    return formatter.vformat(str, (), mapping)

# -------------------------------------
def get_first(mydict, keys):
    """Looks up a series of keys in a dict"""
    for key in keys:
        try:
            return mydict[key]
        except:
            pass
    raise KeyError(keys)
# ----------------------------------------------------
def merge_dicts(*dicts):
    """Merges a bunch of dicts.  Later dicts take precedence over earlier dicts."""
    if len(dicts) == 0:
        return dict()

    mydict = type(dicts[0])(dicts[0].items())
    for xdict in dicts[1:]:
        mydict.update(xdict.items())
    return mydict
# ------------------------------------------------------
# https://lerner.co.il/2014/01/03/making-init-methods-magical-with-autoinit/
def autoinit():
    """Automgagically copy parameters tot __init__() method to self.
    Eg:
        class MyClass(object):
            def __init__(a, b):
                autoinit()
            def do_stuff():
                print(self.a, self.b)
    """
    frame = inspect.currentframe(1)
    params = frame.f_locals
    self = params['self']
    paramnames = frame.f_code.co_varnames[1:] # ignore self
    for name in paramnames:
        setattr(self, name, params[name])
# -----------------------------------------------------------
# https://stackoverflow.com/questions/40367461/intersection-of-two-lists-of-ranges-in-python/40368603
def intersect_ranges(A, B):

    """Given two lists of ranges [low,high), finds the intersection of
    each range in A with each range in B.

    Returns: [(aix, bix, r0, r1), ...]
        One tuple for each intersection.
        aix:
            Index of range in A for which this overlaps.
        bix:
            Index of rnage in B for which this overlaps.
        r0, r1:
            Value of the overlapping range between A[aix] and B[bix]
    """
    return [
        (aix, bix, max(first[0], second[0]), min(first[1], second[1]))
        for aix,first in enumerate(A) for bix,second in enumerate(B)
        if max(first[0], second[0]) <= min(first[1], second[1])]
