import numpy as np


# ------------------------------------------------
def _unnest(obj, compound='_'):
    """Un nest dicts returned by struct_to_dict().

    That is... multiply-nested dicts are turned into a single dict
    with compound keys.

    compound:
        'tuple':
            Make compound tuples
        '<character>':
            Use that as a separator
    """

    if isinstance(obj, list):
        return [_unnest(o) for o in obj]
    is_tuple = (compound == 'tuple')

    # It's a dict
    if isinstance(obj, dict):
        ret = dict()
        for k0,v0 in obj.items():
            val0 = _unnest(v0)

            if isinstance(val0, dict):
                for k1,v1 in val0.items():
                    if is_tuple:
                        if isinstance(k1, tuple):
                            key = tuple([k0] + list(k1))
                        else:
                            key = (k0, k1)
                    else:
                        key = '{}{}{}'.format(k0,compound,k1)
                    ret[key] = v1
            else:
                ret[k0] = v0
        return ret

    # Just pass through
    return obj

# ------------------------------------------------
def structured_to_dict(arr: np.ndarray, unnest=True, compound='_'):

    """Converts nested arrays of structs (recarrays, such as those loaded
    from MATLAB), to nested dicts and arrays.

    https://stackoverflow.com/questions/2203673/efficient-way-to-convert-numpy-record-array-to-a-list-of-dictionary
    Here's a solution that works in the following cases not covered by
    the other answers:

    * 0-dimensional arrays (scalars). e.g. np.array((1, 2), dtype=[('a', 'float32'), ('b', 'float32')])

    * elements of type np.void (result of indexing a record array)

    * multi-dimensional arrays of structs

    * structs containing structs, (e.g. structured_to_dict(np.zeros((), dtype=[('a', [('b', 'float32', (2,))])])))

    * Combinations of any of the above.

    Changes from Original
    ---------------------

    * "Squeezes" multi-dimensional arrays to eliminate
      size-1 dimensions.  This is useful converting MATLAB data
      structures, in which scalars and 1D vectors are often
      represented as degenerate 2D arrays.

    * Recursively processes structures.

    Original Code
    -------------
    ```
    def structured_to_dict(arr: np.ndarray):
        import numpy as np

        if np.ndim(arr) == 0:
            if arr.dtype.names == None:
                return arr.item()
            # accessing by int does *not* work when arr is a zero-dimensional array!
            return {k: structured_to_dict(arr[k]) for k in arr.dtype.names}
        return [structured_to_dict(v) for v in arr]
    ```

    """

    arr = np.squeeze(arr)
    if np.ndim(arr) == 0:
        if arr.dtype.names == None:
            ret = arr.item()
            if isinstance(ret, np.ndarray):
                ret = structured_to_dict(ret)
        else:
            # accessing by int does *not* work when arr is a zero-dimensional array!
            ret = {k: structured_to_dict(arr[k]) for k in arr.dtype.names}
    else:
        ret = [structured_to_dict(v) for v in arr]

    if unnest:
        ret = _unnest(ret, compound=compound)

    return ret
# ------------------------------------------------

#x=structured_to_dict(glas)
#df = pd.DataFrame(_unnest(structured_to_dict(glas)))



#glas = scipy.io.loadmat('data/slater2019/glaciers.mat')['glaciers']
