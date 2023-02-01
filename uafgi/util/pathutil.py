import os

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
    def __init__(self, sep, roots):
        """sep:
            Should be os.sep of the target platform.
        roots:
        """
        self.sep = sep
        self.lookup = dict()
        self.sorted = list()
        self.update(roots)

    def update(self, roots_iter):
        """roots_iter: [(key, root), ...]
            Roots as absolute paths, native to the system they're intended to be used on.
        """
        for key,path in roots_iter:
            print(f'root[{key}] = {path}')
            self.lookup[key] = path
        self.sorted = [(len(val),key) for key,val in self.lookup.items()]
        self.sorted.sort(reverse=True)

    def relpath(self, path_sys):
        """Given a path, converts as relative to a root, AND WITH FORWARD SLASHES.
        This needs to be run on the system native to the path_sys and roots
        path_sys:
            A path native to the system we're running on."""
        path = os.path.abspath(os.path.realpath(path_sys)).replace(os.sep, '/')
        for _,key in self.sorted:
            print('   try {}, {}'.format(path, self.lookup[key]))
            root = self.lookup[key]
            if path.startswith(root):
                return '{'+key+'}' + path[len(root):]
        return path

    def abspath(self, rel):
        """Returns a path native to the system we're runnin on.
        rel:
            Relative path WITH FORWARD SLASHES"""
        rel = rel.replace('/', self.sep)
        path = rel.format(**self.lookup)
        return path

    def bashpath(self, rel):
        """Like abspath, but converts to bash-style pathname"""
        wfname = self.abspath(rel).replace('\\', '/')
        if wfname[1] == ':':
            wfname = '/{}{}'.format(wfname[0], wfname[2:])
        return wfname

    def convert_to(self, path, dest_roots):
        return dest_roots.abspath(self.relpath(path))

#def convert(path, roots0, roots1):
#    """Converts a pathname from convention of roots0 to convention of roots1"""
#    return roots1.abspath(roots0.relpath(path))

    
