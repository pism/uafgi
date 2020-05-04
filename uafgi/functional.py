import types
from uafgi.checksum import hashup
from uafgi import gicollections,giutil
import types
import inspect
import operator
import numpy as np

# Minimal interface for end users and 'from giss.functional import *'
# more involved uses regular imports and more fully qualified namesx
__all__ = (
    '_arg', 'bind', 'function', 'thunkify', 'Function',
    'wrap_value', 'wrap_combine', 'intersect_dicts', 'none_if_different',
    'xtuple', 'xnamedtuple', 'xdict')

# ---------------------------------------------------------
# Universal for all Functions...
#    (f + g)(x) = f(x) + g(x)
class Function(object):
    def __add__(self, other):
        return lift_once(operator.add, self, other)
    def __mul__(self, other):
        return lift_once(operator.mul, self, other)
    def __truediv__(self, other):
        return lift_once(operator.truediv, self, other)
    def __getattr__(self, attr):
        return lift_once(getattr, self, attr)
        
class lift_once(Function):
    """Turns a function on values into a function on functions."""
    def __init__(self, lifted_fn, *fargs, **fkwargs):
        self.lifted_fn = lifted_fn
        self.fargs = fargs    # Arguments are (possibly) functions
        self.fkwargs = fkwargs
    def __call__(self, *args, **kwargs):
        xargs = tuple(
            fn(*args, **kwargs) if isinstance(fn, Function) else fn
            for fn in self.fargs)
        xkwargs = {
            k : v(*args, **kwargs) if isinstance(v,Function) else v
            for k,v in self.fkwargs.items()}

        return self.lifted_fn(*xargs, **xkwargs)

    def __repr__(self):
        return '{}({})'.format(self.lifted_fn, ','.join(repr(x) for x in self.fargs), self.fkwargs)

@giutil.arg_decorator
class lift(Function):
    """Decorator: Turns a function on values into a function on functions."""
    def __init__(self, lifted_fn):
        # TODO: Take more specific args explaining which params are to be dereferenced.
        self.lifted_fn = lifted_fn
    def __call__(self, *args, **kwargs):
        return lift_once(self.lifted_fn, *args, **kwargs)





#class lift(Function):
#    """Decorator, turns a function on values into a function on functions."""
#    def __init__(self, lifted_fn):
#        self.lifted_fn = lifted_fn
#    def __call__(self, *args, **kwargs):
#        args = tuple(
#            fn(*args, **kwargs) if isinstance(fn, Function) else fn
#            for fn in self.funcs)
#        return self.lifted_fn(*args)
#    def __repr__(self):
#        return '{}({})'.format(self.lifted_fn, ','.join(repr(x) for x in self.funcs))



# ---------------------------------------------------------
# Partial binding of functions

class _arg(object):
    """Tagging class"""
    def __init__(self, index, name=''):
        self.index = index
    def __repr__(self):
        return '_arg({})'.format(self.index)

class BoundFunction(Function):
    """Reorder positional arguments.
    Eg: g = f('yp', _1, 17, _0, dp=23)
    Then g('a', 'b', another=55) --> f('yp', 'b', 17, 'a', dp=23, another=55)

    TODO:
       1. When wrapping multiple _Binds, create just a single level???
       2. Get kwargs working
    """

    def __init__(self, fn, *bound_args, **bound_kwargs):
        # Maximum index referred to by the user.
        # Inputs to f above this index will be passed through
        self.fn = fn
        self.bound_args = bound_args
        self.bound_kwargs = bound_kwargs
        self.first_unbound = 1+max(
            (x.index if isinstance(x, _arg) else -1 for x in bound_args),
            default=-1)

    def __call__(self, *gargs, **gkwargs):
        fargs = \
            [gargs[x.index] if isinstance(x, _arg) else x
                for x in self.bound_args] + \
            list(gargs[self.first_unbound:])

        fkwargs = dict(self.bound_kwargs)
        fkwargs.update(gkwargs)    # Overwrite keys
        # print('BEGIN Calling', self.fn)
        ret = self.fn(*fargs, **fkwargs)
        # print('END Calling', self.fn)
        return ret

    hash_version=0
    def hashup(self,hash):
        hashup(hash, (self.fn, self.bound_args, self.bound_kwargs))
    def __repr__(self):
        return 'bind({}, {}, {})'.format(self.fn, self.bound_args, self.bound_kwargs)

def bind(fn, *bound_args, **bound_kwargs):
    if False and isinstance(fn, BoundFunction):
        # (Don't bother with this optimization for now...)
        # Re-work bound args...
        pass
    elif isinstance(fn, _xtuple):
        # Bind inside the xtuple, to retain tuple nature of our function
        return fn.construct(bind(x, *bound_args, **bound_kwargs) for x in fn)
    else:
        return BoundFunction(fn, *bound_args, **bound_kwargs)


def function():
    def real_decorator(python_fn):
        """Decorator Wraps a Python function into a Function."""
        return bind(python_fn)

    return real_decorator
# ---------------------------------------------------------
# Good for summing >1 functions

class sum(Function):
    def __init__(self, *funcs):
        self.funcs = funcs
    def __call__(self, *args, **kwargs):
        sum = funcs[0](*args, **kwargs)
        for fn in funcs[1:]:
            sum += fn(*args, **kwargs)
        return sum

class product(Function):
    def __init__(self, *funcs):
        self.funcs = funcs
    def __call__(self, *args, **kwargs):
        product = funcs[0](*args, **kwargs)
        for fn in funcs[1:]:
            product *= fn(*args, **kwargs)
        return product
# ---------------------------------------------------------
def thunkify():
    class real_decorator(Function):
        """Decorator that replaces a function with a thunk constructor.
        When called, the resulting thunk will run the original function.

        Suppose f :: X -> Y         # Haskell notation
        Then thunkify f x :: Y
        or    thunkfiy :: Function -> X -> Y
        """
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            return bind(self.fn, *args, **kwargs)

        def __repr__(self):
            return 'thunkify({})'.format(self.fn)

    return real_decorator

# -------------------------------------------------------
class wrap_value(Function):
    """A thunk that wraps a value; calling the thunk will return the value.
    This serves as a base class to define arithmetic operations on (wrapped) values
    when combining functions."""
    def __init__(self, value):
        self.value = value
    def __call__(self):
        return self.value
    def __repr__(self):
        return 'wrap_value({})'.format(self.value)

# -------------------------------------------------------------
class wrap_combine(wrap_value):
    """A thunk that wraps a value; calling the thunk will return the value.
    All operations are mapped to a supplied "combine" function."""
    def __init__(self, value, combine_fn):
        self.value = value
        self.combine_fn = combine_fn
    def __add__(self, other):
        other_value = other() if callable(other) else other
        return wrap_combine(self.combine_fn(self.value, other_value), self.combine_fn)
    __mul__ = __add__
    __truediv__ = __add__

# -------------------------------------------------------------
def eq_ndarray(a,b):
    return (a==b).all()

_eq_methods = {np.ndarray : eq_ndarray}

def eq_any(a,b):
    try:
        return _eq_methods[type(a)](a,b)
    except:
        return a==b

# -------------------------------------------------------------
def intersect_keys(a,b):
    """Iterator joins the keys of two dicts"""
    for key in a.keys() & b.keys():
        if eq_any(a[key], b[key]):
            yield key

def intersect_dicts(a,b):
    """Combine function: Returns only entries with the same value in both dicts."""
    if not isinstance(a, dict):
        return b
    if not isinstance(b, dict):
        return a

    ret = type(a)()
    for key in a.keys() & b.keys():
        if eq_any(a[key], b[key]):
            ret[key] = a[key]
    return ret

#        print('xxxxxxx', key, type(a[key]), type(b[key]))
#        print('       ', a[key] == b[key])

#    keyvals = {key : a[key] \
#        for key in a.keys() & b.keys() \
#        if a[key] == b[key]}
#
#    return type(a)(keyvals)

def none_if_different(a,b):
    """Combine function: Keep only if things are the same."""
    return a if a==b else None
# -------------------------------------------------------------
class _xtuple(Function):
    """Avoid problems of multiple inheritence and __init__() methods.
    See for another possible soultion:
    http://stackoverflow.com/questions/1565374/subclassing-python-tuple-with-multiple-init-arguments"""

    def __call__(self, *args, **kwargs):
        return self.construct(x(*args, **kwargs) for x in self)

    def _map2_(self, fn, other):
        if isinstance(other, _xtuple):
            return self.construct(fn(s,o) for s,o in zip(self,other))
        else:
            return self.construct(fn(s,other) for s in self)

    def __add__(self, other):
        return self._map2_(operator.add, other)
    def __mul__(self, other):
        return self._map2_(operator.mul, other)
    def __truediv__(self, other):
        return self._map2_(operator.truediv, other)


class xtuple(_xtuple,tuple):
    def construct(self, args):
        """Construct a new instance of type(self).
        args:
            A (Python)tuple of the things to construct with."""
        return type(self)(args)
    def __repr__(self):
        return '(x ' + ','.join(repr(x) for x in self) + ')'

class _NamedxTupleBase(_xtuple,tuple):
    """For use only with xnamedtuple.  Accomodates different constructors
    for tuple vs namedtuple."""
    def construct(self, args):
        """Construct a new instance of type(self).
        args:
            A (Python)tuple of the things to construct with."""
        return type(self)(*args)

def xnamedtuple(*args, **kwargs):
    return gicollections.namedtuple(*args, tuple_class=_NamedxTupleBase, **kwargs)
# -----------------------------------------------------------------
class xdict(Function,dict):
    def _join(self, other):
        """Generator that lists the keys in common"""
        if isinstance(other, dict):
            for key in self:
                if key in other:
                    yield key,self[key],other[key]
        else:
            for key in self:
                yield key,self[key],other


    def __add__(self, other):
        return xdict({k : s + o for k,s,o in self._join(other)})

    def __mul__(self, other):
        return xdict({k : s * o for k,s,o in self._join(other)})

    def __truediv__(self, other):
        return xdict({k : s / o for k,s,o in self._join(other)})

    def __call__(self, *args, **kwargs):
        return xdict({k : v(*args,**kwargs) \
            for k,v in self._items()})

    def __getitem__(self, key):
        return xdict({k : v[key] \
            for k,v in self._items()})

    def __getattr__(self, attr):
        return xdict({k : getattr(v, attr) \
            for k,v in self._items()})

    def _getitem(self, key):
        return dict.__getitem__(self, key)
    def _items(self):
        return dict.items(self)
    def _keys(self):
        return dict.keys(self)
    def _values(self):
        return dict.values(self)
    def _update(self, other):
        for k,v in other._items():
            self[k] = v

#class xattrdict(xdict):
#    def __getattr__(self, name):
#        return self[name]
# --------------------------------------------------
        
