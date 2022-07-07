import os


def root(dir0):
    """Given a directory within a git checkout, finds the root of the
    checkout"""

    dir0 = os.path.abspath(dir0)
    dir = dir0
    FSROOT = os.path.abspath(os.sep)
    while dir != FSROOT:
        if os.path.exists(os.path.join(dir, '.git')):
            return dir
        dir = os.path.split(dir)[0]

    raise ValueError(f'Directory {dir0} is not in a git tree')


def harness_root(dir):
    """Given a directory within a harness, finds the harness (one
    directory above a git checkout dir)."""

    return os.path.split(root(dir))[0]

