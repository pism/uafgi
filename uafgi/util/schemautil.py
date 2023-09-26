import yaml
import schema

# -----------------------------------------------
class Quantity(schema.Use):
    def __init__(self, units, description, error=None):
        super().__init__(float, error=error)
        self.format = repr
        self.description = description
        self.units = units

# -----------------------------------------------
class Field(schema.Use):
    def __init__(self, parse_fn, format_fn, description, error=None):
        super().__init__(parse_fn, error=error)
        self.format = format_fn
        self.description = description
        self.units = None

class Int(schema.Use):
    def __init__(self, description, error=None):
        super().__init__(int, error=error)
        self.format = repr
        self.description = description
        self.units = None

def _nullable_int(s):
    if s is None or s == '.':
        return None
    return int(s)

class NullableInt(schema.Use):
    def __init__(self, description, error=None):
        super().__init__(_nullable_int, error=error)
        self.format = repr
        self.description = description
        self.units = None

# -----------------------------------------------
_bool_values = {
    '0': False, '1': True,
    'false': False, 'true': True,
    'f': False, 't': True,
    'no': False, 'yes': True,
    'n': False, 'y': True,
}
def _parse_bool(sbool):
    return _bool_values[sbool.lower()]

class Bool(schema.Use):
    def __init__(self, description, error=None):
        super().__init__(_parse_bool, error=error)
        self.format = repr
        self.description = description
        self.units = None

# -----------------------------------------------
class InputFile(schema.Use):
    def __init__(self, roots, description, error=None):
        """roots: pathutil.RootsDict
        """
        self.roots = roots    # Howto interpret file
        self.description = description
        self.units = units

    def validate(self, relpath, **kwargs):
        """data:
            Abstract path, eg: {DATA}/mystuff/stuff.nc
        """
        syspath = self.roots.syspath(relpath)
        if not os.path.exists(syspath):
            raise FileNotFoundError(syspath)
        return syspath

    def format(self, syspath):
        return self.roots.relpath(syspath)
# -----------------------------------------------
class ParsedEnumField(schema.Use):
    def __init__(self, parse_fn, format_fn, enums, description):
        self.parse = parse_fn
        self.format = format_fn
        self._enums = set(enums)
        self._sorted_enums = sorted(enums)
        self.description = f'{description}.  Legal values are: {enums}'
        self.units = None

    def __repr__(self):
        return repr(self._sorted_enums)

    def validate(self, data, **kwargs):
        parsed = self.parse(data)
        if parsed in self._enums:
            return parsed
        raise schema.SchemaError(f"{repr(parsed)} not in enumeration {repr(self)}")
# -----------------------------------------------
class EnumField(schema.Use):
    def __init__(self, enums, description):
        self._enums = set(enums)
        self._sorted_enums = sorted(enums)
        self.description = f'{description}.  Legal values are: {enums}'
        self.units = None

    def __repr__(self):
        return repr(self._sorted_enums)

    def validate(self, data, **kwargs):
        if data in self._enums:
            return data
        raise schema.SchemaError(f"{repr(data)} not in enumeration {repr(self)}")

    def format(self, data):
        return data
# -----------------------------------------------
class DictField(schema.Use):
    def __init__(self, map, description):
        self._map = map
        self._invmap = {v:k for k,v in map.items()}
        self._sorted_keys = sorted(map.keys())
        self.description = f'{description}.  Legal values are: {enums}'
        self.units = None

    def __repr__(self):
        return repr(self._sorted_keys)

    def validate(self, data, **kwargs):
        try:
            return self._map[data]
        except KeyError:
            raise schema.SchemaError(f"{repr(data)} not in mapped enumeration {repr(self)}") from None

    def format(self, data):
        return self._invmap[data]

# -----------------------------------------------
# Access Schemas...
# -----------------------------------------------
def _ident(x):
    return x

def schema_getitem(schema, key):
    """Looks up a field in a schema by name.
    Returns: value (the field)"""

    for keyfn in (_ident, schema.Optional, schema.Hook):
        try:
            return schema.schema[keyfn(key)]
        except KeyError:
            pass

    raise KeyError(key)

def schema_items(schema):
    for key,val in schema.schema.items():
        if isinstance(key, str):
            yield key,val
        else:
            yield key.schema,val    # This actually gives the name of the key

def format(vals, schema=None):
    """Reverses the validation (parsing) of values in a schema."""
    for k,v in vals.items():
        field = schema.schema[k]

