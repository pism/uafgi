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

    def __setitem__(self, key, val):
        self.lookup[key] = val

    def update(self, roots_iter):
        """roots_iter: [(key, root), ...]
            Roots as absolute paths, native to the system they're intended to be used on.
        """
        for key,path in roots_iter:
            print(f'root[{key}] = {path}')
            self.lookup[key] = path
        self.sorted = [(len(val),key) for key,val in self.lookup.items()]
        self.sorted.sort(reverse=True)

    def relpath(self, syspath):
        """Given a path, converts as relative to a root, AND WITH FORWARD SLASHES.
        This needs to be run on the system native to the syspath and roots
        syspath:
            A path native to the system we're running on."""
        path = os.path.abspath(os.path.realpath(syspath)).replace(os.sep, '/')
        for _,key in self.sorted:
            root = self.lookup[key]
            if path.startswith(root):
                return '{'+key+'}' + path[len(root):]
        return path

    def syspath(self, rel, bash=False):
        """Returns a path native to the system we're running on.
        rel:
            Relative path WITH FORWARD SLASHES"""

        if bash:
            path = rel.format(**self.lookup).replace('\\', '/')
            if path[1] == ':':
                path = '/{}{}'.format(path[0], path[2:])
            return path
        else:
            rel = rel.replace('/', self.sep)
            path = rel.format(**self.lookup)
            return path


    def join(self, *args, bash=False):
        """For compatibility with old dggs.data.join()"""
        path = list(args)
        path[0] = '{'+args[0].upper()+'}'
        relpath = '/'.join(path)
        return self.syspath(relpath, bash=bash)
        

    def convert_to(self, syspath, dest_roots, bash=False):
        """syspath:
            A path native to the source system
        dest_roots:
            Path configuration for the remote system.
        """
        return dest_roots.syspath(self.relpath(syspath), bash=bash)
