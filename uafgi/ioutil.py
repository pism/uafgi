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

import contextlib
import sys
import os
import re
import string
import tempfile
import filecmp
import shutil

# http://stackoverflow.com/questions/13250050/redirecting-the-output-of-a-python-function-from-stdout-to-variable-in-python
@contextlib.contextmanager
def redirect(out=sys.stdout, err=sys.stderr):
    """A context manager that redirects stdout and stderr"""
    saved = (sys.stdout, sys.stderr)
    sys.stdout = out
    sys.stderr = err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved

@contextlib.contextmanager
def pushd(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(prev_cwd)


def list_dir(logdir, regexp, key=lambda match: match.group(0)):
    """List the files in a directory that match a regexp
    logdir:
        Directory to list files in.
    regexp: str
        The regular expression to match.
    key: lambda match
        Key function, called key(match) on matching records.
        This allows extraction of filename stuff, etc.
        By default, returns the original filename.
    returns: [(key, fname)]
        key: Result of running the key function on the file
        fname: Full filename (including logdir)"""
    regexpRE = re.compile(regexp)
    fnames = []
    for leaf in os.listdir(logdir):
        match = regexpRE.match(leaf)
        if match is not None:
            fnames.append((key(match), os.path.join(logdir, leaf)))
    return sorted(fnames)

class AtomicOverwrite(object):
    """Writes a file, swapping it to overwrite the previous file atomically"""
    def __init__(self, name, mode='w'):
        self.name = name    # Filename
        self.tmp = self.name + '.tmp'
        self.mode = mode
        self.out = None

    def __enter__(self):
        self.out = open(self.tmp, self.mode)
        return self

    def __exit__(self, *args):
        """Default is to NOT commit."""
        if self.out is not None:
            self.out.close()
            self.out = None

    def commit(self):
        """If user calls commit(), THEN we commit."""
        self.__exit__()
        os.rename(self.tmp, self.name)


class WriteIfDifferent(object):
    """Writes a file, swapping it to overwrite the previous file atomically"""
    def __init__(self, name, mode='w'):
        self.name = name    # Filename
        self.tmp = self.name + '.tmp'
        self.mode = mode
        self.out = None

    def __enter__(self):
        self.out = open(self.tmp, self.mode)
        return self

    def rollback(self):
        os.remove(self.tmp)

    def __exit__(self, *args):
        # Close the file
        if self.out is not None:
            self.out.close()
            self.out = None

        # Compare to
        if (not os.path.exists(self.name)):
            # Writing a virgin file
            os.rename(self.tmp, self.name)
        else:
            # Compare contents to what's there
            with open(self.name, 'rb') as old_file:
                old_contents = old_file.read()
            with open(self.tmp, 'rb') as new_file:
                new_contents = new_file.read()

            if old_contents == new_contents:
                self.rollback()
            else:
                print('Writing {}'.format(self.name))
                os.remove(self.name)
                os.rename(self.tmp, self.name)

    def close(self):
        self.__exit__()

#class WriteIfDifferent(object):
#    """Allows user to write to a temporary file, then move it
#    to the destination only if it is different from the destination."""
#    def __init__(self, ofname, **kwargs):
#        """ofname: Name we ultimately want to write to."""
#        self.ofname = ofname
#        self.out = tempfile.NamedTemporaryFile(delete=False, **kwargs)
#        self.tfname = self.out.name
#
#    def __enter__(self):
#        pass
#
#    def close(self):
#        self.__exit__()
#
#    def rollback(self):
#        self.out.close()
#        os.remove(self.tfname)
#
#    def __exit__(self, *args):
#        self.out.close()
#        try:
#            if filecmp.cmp(self.tfname, self.ofname):
#                # Files are equal, we are done!
#                os.remove(self.tfname)
#                return
#        except: pass # Error means the files were NOT equal.
#
#        # Files are not equal, so copy the temporary file over.
#        shutil.copyfile(self.tfname, self.ofname)
#        os.remove(self.tfname)



def needs_regen(ofiles, ifiles):
    """Determines if any of the ofiles are older than any of the ifiles.
    This is used, eg in make, to determine if a ruile needs to be run."""

    try:
        otimes = [os.path.getmtime(x) for x in ofiles]
    except FileNotFoundError:
        return True

    # It's an error if the input files don't all exist.
    itimes = [os.path.getmtime(x) for x in ifiles]

    min_otime = min(otimes)
    max_itime = max(itimes)

    return max_itime >= min_otime

# http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
def search_file(filename, search_path):
     """Given a search path, find file
     """
     file_found = 0
     paths = search_path.split(os.pathsep)
     for path in paths:
         if os.path.exists(os.path.join(path, filename)):
             file_found = 1
             break
     if file_found:
         return os.path.abspath(os.path.join(path, filename))
     else:
         return None

class TmpFiles(object):
    def __init__(self, root):
        self.root = root
        self.next_tmp = 0

    def __next__(self):
        """Get a tmp filename"""
        ret = '{}_{}'.format(self.root, self.next_tmp)
        self.next_tmp += 1
        return ret

    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, exc_traceback):
        for i in range(0,self.next_tmp):
            try:
                os.remove('{}_{}'.format(self.root, i))
            except FileNotFoundError:
                pass





