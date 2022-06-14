import hashlib
import struct
import types


__all__ = ('begin_checksum', 'hashup', 'checksum')

# Our chosen hash function
begin_checksum = hashlib.md5

"""To make a class checksummable:

   1. Add tag hash_version, which you will change when you want to
      change the hash.
   2. Add as hashup() method.

   Eg:
   class MyClass(object):
       hash_version = 17
       def __init__(self):
           self.val = 17
       def hashup(self, hash):
           hashup(hash, self.val)
"""
# --------------------------------------------------------
def hashup_int(hash, x):
    try:
        # https://docs.python.org/2/library/struct.html#format-characters
        bytes = struct.pack('>i',x)
    except struct.error:    # x is too large
        bytes = str(n).encode()
    hash.update(bytes)

def hashup_float(hash, x):
    hash.update(struct.pack('>f',x))

def hashup_bool(hash, x):
    bytes = struct.pack('>i',1 if x else 0)

def hashup_str(hash, x):
    hash.update(x.encode())

def hashup_bytes(hash, x):
    hash.update(x)

def hashup_sequence(hash, coll):
    for x in coll:
        hashup(hash, x)

def hashup_set(hash, coll):
    hashup_sequence(sorted(tuple(coll)))

def hashup_dict(hash, coll):
    hashup_sequence(hash, sorted(tuple(coll.items())))

def hashup_fn(hash, fn):
    hash.update(fn.__module__.encode())
    hash.update(fn.__qualname__.encode())  # https://www.python.org/dev/peps/pep-3155/

def hashup_method(hash, method):
    hashup(hash, method.__self__)
    hash.update(method.__module__.encode())
    hash.update(method.__qualname__.encode())  # https://www.python.org/dev/peps/pep-3155/


def hashup_module(hash, mod):
    hash.update(mod.__package__.encode())
    hash.update(mod.__name__.encode())

def hashup_type(hash, klass):
    hash.update(klass.__module__.encode())
    hash.update(klass.__qualname__.encode())


# -----------------------------------------
def hashup_error(hash, x):
    raise ValueError('Cannot checksum {} {}'.format(type(x), x))

hashup_methods = {
    int : (b'int', hashup_int),
    bool : (b'bool', hashup_bool),
    float : (b'float', hashup_float),
    str : (b'str', hashup_str),
    bytes : (b'bytes', hashup_bytes),
    tuple : (b'tuple', hashup_sequence),
    list : (b'list', hashup_sequence),
    set : (b'set', hashup_set),
    dict : (b'dict', hashup_dict),
    type : (b'type', hashup_type),
    types.FunctionType : (b'types.FunctionType', hashup_fn),
    types.MethodType : (b'types.MethodType', hashup_method),
    types.GeneratorType : (b'types.GeneratorType', hashup_fn),
    types.CoroutineType : (b'types.CoroutineType', hashup_fn),
    types.BuiltinFunctionType : (b'types.BuiltinFunctionType', hashup_fn),
    types.BuiltinMethodType : (b'types.BuiltinMethodType', hashup_fn),
    types.ModuleType : (b'types.ModuleType', hashup_module),
}

def hashup(hash, x, klass=None):
    if x is None:
        hash.update(b'__NONE__')
        return

    klass = type(x)
    if klass in hashup_methods:
        tag, hashup_fn = hashup_methods[klass]
        hash.update(tag)
        hashup_fn(hash, x)
    else:
        hash.update(b'object')
        hash.update(klass.__module__.encode())
        hash.update(klass.__name__.encode())
        hashup(hash, klass.hash_version)
        x.hashup(hash)

def checksum(x):
    """Top-level function"""
    hash = begin_checksum()
    hashup(hash, x)
    return hash.digest()
