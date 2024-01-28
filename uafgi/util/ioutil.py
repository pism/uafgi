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

import contextlib,pathlib
import sys
import os,subprocess
import re
import string
import tempfile
import filecmp
import shutil
import signal,functools

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
    """A context manager that changes the working directory to the given
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
        self.name = pathlib.Path(name)    # Final Filename
        self.tmpfile = self.name.parents[0] / (self.name.parts[-1] + '.tmp')

    def __enter__(self):
        return self

    def rollback(self):
        os.remove(self.tmpfile)

    def __exit__(self, *args):
        # Compare to
        if (not os.path.exists(self.name)):
            # Writing a virgin file
            os.rename(self.tmpfile, self.name)
        else:
            # Compare contents to what's there
            with open(self.name, 'rb') as old_file:
                old_contents = old_file.read()
            with open(self.tmpfile, 'rb') as new_file:
                new_contents = new_file.read()

            if old_contents == new_contents:
                self.rollback()
            else:
                print('Writing {}'.format(self.name))
                os.remove(self.name)
                os.rename(self.tmpfile, self.name)

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

class TmpDir(object):
    """Context manager that creates a temporary directory, which will be
    removed upon exit, or even Ctrl-C.  The caller can put files in
    the temporary directory that is created; and they will all
    disappear upon exit.
    """

    def __init__(self, dir='.', remove=True, tdir=None, clear=False):
        """
        dir:
            Directory in which to create the temporary directory.
            (Useful if large files are to be created, this should be in
             the same filesystem as the ultimate resulting file will be)
        remove:
            Remove the temporary directory when done?
            Set to False for debugging, to see what happened.
        tdir:
            Name of the temporary directory to use directly.
            If this is set, then dir and remove will be ignored.
            The user-defined tdir director will NOT be removed.
            For debugging into a consistently named directory...
        clear:
            Clear contents of the directory if it already exists?
        """
        self.tdir = tdir
        if self.tdir is not None:
            self.remove = False    # Don't remove user-specified tdir
        else:
            self.dir = dir
            self.remove = remove
        self.clear = clear

    @property
    def location(self):
        return self.tempd

    def _handler(self, sig, frame):
        """Called when user does Ctrl-C (SIGINT)"""
        self._remove()
        self.original_sigint_handler(sig, frame)

    def __enter__(self):
        if self.tdir is not None:
            if self.clear:
                shutil.rmtree(self.tdir)
            os.makedirs(self.tdir, exist_ok=True)
            self.tempd = self.tdir
        else:
            self.tempd = tempfile.mkdtemp(dir=self.dir)
            if not self.remove:
                print('Creating temporary directory {}'.format(self.tempd))

            # https://stackoverflow.com/questions/22916783/reset-python-sigint-to-default-signal-handler
            self.original_sigint_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self._handler)
        return self

    def _remove(self):
        if self.remove:
            signal.signal(signal.SIGINT, self.original_sigint_handler)
            shutil.rmtree(self.tempd, ignore_errors=True)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._remove()

    # -------------------- Uses for tdir
    def subdir(self, **kwargs):
        """Create a temporary subdirectory."""
        return TmpDir(dir=self.tempd, **kwargs)

    def join(self, *args):
        """Produces a file with a specific name inside the tdir"""
        return os.path.join(self.tempd, *args)

    text_by_mode = {
        'r' : True,
        'rt' : True,
        'rb' : False}
    def open(self, suffix=None, prefix=None, mode='rt'):
        """Creates a temporary file in the most secure manner possible. There
        are no race conditions in the file’s creation, assuming that
        the platform properly implements the os.O_EXCL flag for
        os.open(). The file is readable and writable only by the
        creating user ID. If the platform uses permission bits to
        indicate whether a file is executable, the file is executable
        by no one. The file is not inherited by child
        processes.

        suffix:
            Suffix of filename.  If suffix is not None, the file name
            will end with that suffix, otherwise there will be no
            suffix. mkstemp() does not put a dot between the file name
            and the suffix; if you need one, put it at the beginning
            of suffix.

        prefix:
            If prefix is not None, the file name will begin with that
            prefix; otherwise, a default prefix is used. The default
            is the return value

        mode: 'r', 'rt' or 'rb'
            File open mode, commensurate with Python IO's open()
            'r', 'rt' = Open text file
            'rb' = Open binary file
        Returns:
            Python file object, as if opened with open()
        """
        handle,path = tmpfile.mkstemp(suffix=suffix, prefix=prefix, dir=self.tempd,
            text=self.text_by_mode[mode])
        return os.fdopen(handle, mode='rt' if text else 'rb')


    def filename(self, suffix=None, prefix=None, mode='rt'):
        """Produces a filename"""
        handle,path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=self.tempd,
            text=self.text_by_mode[mode])
        os.close(handle)
        return path


    def opath(self, ipath, suffix, replace=None):
        """Calls make.opath; to create new filename in temporary dir, based
        on old filename.

        Converts from [idir]/[ipath][ext] to [odir]/[opath][suffix][ext]
        ipath:
            Full pathname of input file
        """    
        idir,ileaf = os.path.split(ipath)
        iroot,iext = os.path.splitext(ileaf)
        if replace is None:
            leaf = '{}{}{}'.format(iroot,suffix,iext)
        else:
            leaf = '{}{}'.format(iroot.replace(replace, suffix), iext)
        return os.path.join(self.tempd, leaf)




# # Also see:
# def temporaryFilename(prefix=None, suffix='tmp', dir=None, text=False, removeOnExit=True):
#     """Returns a temporary filename that, like mkstemp(3), will be secure in
#     its creation.  The file will be closed immediately after it's created, so
#     you are expected to open it afterwards to do what you wish.  The file
#     will be removed on exit unless you pass removeOnExit=False.  (You'd think
#     that amongst the myriad of methods in the tempfile module, there'd be
#     something like this, right?  Nope.)"""
# 
#     if prefix is None:
#         prefix = "%s_%d_" % (os.path.basename(sys.argv[0]), os.getpid())
# 
#     (fileHandle, path) = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir, text=text)
#     os.close(fileHandle)
# 
#     def removeFile(path):
#         os.remove(path)
#         logging.debug('temporaryFilename: rm -f %s' % path)
# 
#     if removeOnExit:
#         atexit.register(removeFile, path)
# 
#     return path
# 

def readlink(fname):
    """Reads a symlink, returning the absolute path it points to."""
    ofname = os.readlink(fname)
    if os.path.isabs(ofname):
        return ofname
    return os.path.abspath(os.path.join(os.path.split(fname)[0], ofname))

def setlink(ifname, ofname):
    print('setlink ifname: {}'.format(ifname))
    print('setlink ofname: {}'.format(ofname))

    if os.path.islink(ofname):
        if os.path.abspath(readlink(ofname)) == os.path.abspath(ifname):
            return    # Already set correctly

        # It was pointing to the wrong thing, remove the link
        print('Unlinking: {}'.format(ofname))
        os.unlink(ofname)

    elif os.path.exists(ofname):
        if not os.path.islink(ofname):
            raise ValueError("Trying to set link target {}, it is currently not a link".format(ofname))

    # Create output directory
    odir,_ = os.path.split(ofname)
    os.makedirs(odir, exist_ok=True)

    # Set the link!
#    print(f'ifname: {ifname}')
#    print(f'')
    os.symlink(os.path.relpath(ifname, start=os.path.split(ofname)[0]), ofname)

def mkdirs_for_files(files):
    """Creates all the directories needed to store the files.
    files: [str, ...]
        Files whose directories we must create.
    """

    # Create directories needed for output files
    dirs = set(os.path.split(x)[0] for x in files)
    for dir in dirs:
        print(f'Creating directory: {dir}')
        os.makedirs(dir, exist_ok=True)

@functools.lru_cache()
def _listdir(dir):
    return sorted(os.listdir(dir))

def matchdir(dir, sregex):
    regex = re.compile(sregex)
    ret = list()
    for file in _listdir(dir):
        match = regex.match(file)
        if match is not None:
            ret.append(match)
    return ret


def sync_files(dir0, fnames, dir1, flags=['--copy-links', '-avzr']):
    """Syncs a list of files into the same location in the remote harness.

    dir0:
        Source directory
    fnames:
        Filenames inside of dir0
        (os.path.relpath() will be used to get relative paths)
    dir1:
        Destination directory
    """

    # Get names of the files, relative to the harness
    fnames_rel = [os.path.relpath(x, dir0) for x in fnames]

    # Write the names to a file contain a list of filenames
    os.makedirs(dir1, exist_ok=True)
    list_file = os.path.join(dir1, '_files.txt')
    with open(list_file, 'w') as out:
        out.write('\n'.join(fnames_rel))
        out.write('\n')

    # Run rsync
    # Create output directory
    # NOTE: As per rsync man page, -r is normally implied by -a; but
    # here it is needed because --files-from is used.
    cmd = ['rsync'] + flags + ['--files-from={}'.format(list_file),
        dir0, dir1]    # No remote hosts in this rsync

    print(cmd)
    subprocess.run(cmd)




# [1]+  Stopped                 jupyter notebook --no-browser --port 1234
