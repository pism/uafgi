import os,pathlib

# https://stackoverflow.com/questions/6718196/determine-prefix-from-a-set-of-similar-strings
# Return the longest prefix of all list elements.
def commonprefix(m):
    "Given a list of pathnames, returns the longest common leading component"
    if not m: return ''
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1



class RootsDict:
    """Manage conversion between full pathnames and user-provided root
    directories."""
    def __init__(self, PureSysPath, roots):
        """PureSysPath:
            Class to use in constructing system path for this type of system.
        roots:
        """
        assert not isinstance(PureSysPath, str)
        self.PureSysPath = PureSysPath
        self.lookup = dict()    # key -> self.PureSysPath
        self.sorted = list()
        self.update(roots)

    def __setitem__(self, key, val):
        self.lookup[key] = self.PureSysPath(val)

    def __getitem__(self, key):
        return self.lookup[key]

    def update(self, roots_iter):
        """roots_iter: [(key, root), ...]
            Roots as absolute paths, native to the system they're intended to be used on.
        """
        for key,path in roots_iter:
            print(f'root[{key}] = {path}')
            self.lookup[key] = self.PureSysPath(path)
        self.sorted = [(len(str(val)),key) for key,val in self.lookup.items()]
        self.sorted.sort(reverse=True)

#    def relpath_key(self, syspath, key):
#        """Given a path, converts as relative to a specified root, AND WITH FORWARD SLASHES.
#        This needs to be run on the system native to the syspath and roots
#        syspath:
#            A path native to the system we're running on.
#        """
#        path = os.path.abspath(os.path.realpath(syspath))#.replace(os.sep, '/')
#        root = self.lookup[key]
#        if root in path.parents:
#            return pathlib.PurePosixPath(path.relative_to(root))
#        raise ValueError(f'The path {syspath} must start with {root}')

    def relpath(self, syspath):
        """Given a path, converts as relative to a root, AND WITH FORWARD SLASHES.
        This needs to be run on the system native to the syspath and roots
        syspath:
            A path native to the system we're running on."""
        path = pathlib.Path(syspath).resolve()
#        path = os.path.abspath(os.path.realpath(syspath))#.replace(os.sep, '/')
        for _,key in self.sorted:
            root = self.lookup[key]
            if root in path.parents:
                return pathlib.PurePosixPath('{'+key+'}') / path.relative_to(root)
        return pathlib.PurePosixPath(path)


    def syspath(self, rel, bash=False):
        """Returns a path native to the system we're running on.
        rel: pathlib.PurePosixPath
            Relative path WITH FORWARD SLASHES"""

        rel = pathlib.PurePosixPath(rel)
        if bash:
            # Add initial stem as posix path
            part0 = rel.parts[0].format(**self.lookup)
            print('ppppppppppp ', part0)

            # Convert drive letter
            part0 = path0.parts[0]
            if part0[1] == ':':
                part0 = '/' + part0[0]

            # Assemble as PurePosixPath (for bash)
            return pathlib.PurePosixPath(part0) / path0.parts[1:] / rel.parts[1:]

        else:
            path0 = self.PureSysPath(rel.parts[0].format(**self.lookup))
            return path0.joinpath(*rel.parts[1:])


    def join(self, *args, bash=False):
        """For compatibility with old dggs.data.join()"""
        part0 = self.lookup[args[0].upper()]
        return self.PureSysPath(part0, *args[1:])
        

    def convert_to(self, syspath, dest_roots, bash=False):
        """syspath:
            A path native to the source system
        dest_roots:
            Path configuration for the remote system.
        """
        return dest_roots.syspath(self.relpath(syspath), bash=bash)
