import os.path

class PFile(dict):
    """Base class: Dict that holds parsed filename parts"""

    @property
    def root(self):
        """The directory of the file"""
        return self['dir']

    @property
    def leaf(self):
        """The leafname"""
        return self.format()

    @property
    def path(self):
        """Full pathname"""
        return os.path.join(self['dir'], self.leaf)
# --------------------------------------------------------
def filter_attrs(attrs):
    """Filteres a pfile if all the given attrs match"""
    def fn(pfile):
        # See if it matches attrs
        match = True
        for (k,v) in attrs.items():
            if pfile[k] != v:
                return False
        return True
    return fn

def listdir(dir, parser_fn, filter_fn):
    """Lists files in a directory, that match the attributes
    Yields parsed dict output"""

    pfiles = list()
    for leaf in os.listdir(dir):
        pfile = parser_fn(os.path.join(dir, leaf))
        if pfile is None:
            continue

        if filter_fn(pfile):
            pfiles.append(pfile)

    pfiles.sort(key=type(pfiles[0]).key_fn)

    return pfiles
# --------------------------------------------------------------
