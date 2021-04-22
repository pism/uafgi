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

import numpy as np
import netCDF4
import sys
import os
import shutil
import subprocess
import contextlib
from uafgi import gicollections

# Copy a netCDF file (so we can add more stuff to it)
class copy_nc(object):
    def __init__(self, nc0, ncout,
        attrib_filter = lambda x : x != '_FillValue'):
        """attrib_filter : function(attrib_name) -> bool
            Only copy attributes where this filter returns True."""
        self.nc0 = nc0
        self.ncout = ncout
        self.attrib_filter = attrib_filter
        self.avoid_vars = set()
        self.avoid_dims = set()
        self.vars = []

    def createDimension(self, dim_name, *args, **kwargs):
        self.avoid_dims.add(dim_name)
        return self.ncout.createDimension(dim_name, *args, **kwargs)

    def copyDimensions(self, *dim_names):
        for dim_name in dim_names:
            l = len(self.nc0.dimensions[dim_name])
            self.ncout.createDimension(dim_name, l)

    def createVariable(self, var_name, *args, **kwargs):
        self.avoid_vars.add(var_name)
        return self.ncout.createVariable(var_name, *args, **kwargs)

    def define_vars(self, _var_pairs, **kwargs):
        """
        kwargs:
            Arguments supplied to NetCDF4 createVariable()
            Use to compress, etc.
        """

        # Standardize var_pairs
        var_pairs = []
        for vp in _var_pairs:
            if isinstance(vp, str):
                var_pairs.append((vp,vp))
            else:
                var_pairs.append(vp)

        self.vars += var_pairs

        # Figure out which dimensions to copy
        copy_dims = set()
        for ivname,ovname in var_pairs:
            for dimname in self.nc0.variables[ivname].dimensions:
                copy_dims.add(dimname)

        # Copy the dimensions!
        for dimname in copy_dims:
            if dimname not in self.ncout.dimensions:
                extent = len(self.nc0.dimensions[dimname])
                self.ncout.createDimension(dimname, extent)

        # Define the variables
        for ivname,ovname in var_pairs:
            var = self.nc0.variables[ivname]
            kwargs1 = dict(kwargs.items())
            if hasattr(var, '_FillValue'):
                FillValue = var._FillValue
                kwargs1['fill_value'] = FillValue
            varout = self.ncout.createVariable(ovname, var.dtype, var.dimensions, **kwargs)
            for aname in var.ncattrs():
                if not self.attrib_filter(aname) : continue
                setattr(varout, aname, getattr(var, aname))

    def define_all_vars(self, **kwargs):
        var_pairs = self.nc0.variables.keys()
        self.define_vars(var_pairs, **kwargs)

    def copy_var(self, ivname, ovname=None):
        if ovname is None:
            ovname = ivname

        print('Copying {}'.format(ivname))
        ivar = self.nc0.variables[ivname]
        ovar = self.ncout.variables[ovname]
        ovar[:] = ivar[:]

    def copy_data(self):
        # Copy the variables
        for ivname,ovname in self.vars:
            self.copy_var(ivname,ovname)


def default_diff_fn(var, val0, val1):
    """Called when we see a difference"""
    pass

def diff(nc0, nc1, ncout=None,
        var_filter=lambda x : x,
        rtol=None, atol=None, equal_nan=None,    # np.isclose()
         **kwargs):    # nc.createVariable()
    """Finds differences between two NetCDF files.
    Optional args: rtol, atol, equal_nan.  See nump.isclose()"""

    isclose_kwargs = dict()
    if rtol is not None:
        isclose_kwargs['rtol'] = rtol
    if atol is not None:
        isclose_kwargs['atol'] = atol
    if equal_nan is not None:
        isclose_kwargs['equal_nan'] = equal_nan


    opened = list()

    try:
        if not isinstance(nc0, netCDF4.Dataset):
            nc0 = netCDF4.Dataset(nc0, 'r')
            opened.append(nc0)

        if not isinstance(nc1, netCDF4.Dataset):
            nc1 = netCDF4.Dataset(nc1, 'r')
            opened.append(nc1)

        if ncout is not None:
            if not isinstance(ncout, netCDF4.Dataset):
                ncout = netCDF4.Dataset(ncout, 'w', clobber=True)
                opened.append(ncout)

        extra0 = list()
        extra1 = list()
        diffs = list()

        remain1 = set([var for var in nc1.variables if var_filter(var) is not None])
        vars0 = [var for var in nc0.variables if var_filter(var) is not None]

        for var in vars0:
            if var not in remain1:
                extra0.append(var)
            else:
                val0 = nc0.variables[var][:]
                val1 = nc1.variables[var][:]
                if not np.isclose(val0, val1, **isclose_kwargs).all():
                    diffs.append(var)

            remain1.remove(var)

        for var in remain1:
            extra1.append(var)

        # Write out if we're given an output file
        diffs_set = set(diffs)
        if ncout is not None:
            nccopy = copy_nc(nc0, ncout,
                var_filter=lambda x : x if x in diffs_set else None)
            nccopy.define_vars(**kwargs)

            for var in diffs:
                val0 = nc0.variables[var][:]
                val1 = nc1.variables[var][:]
                ncout[var][:] = val1 - val0


        return (extra0, extra1, diffs)

    finally:
        for nc in opened:
            try:
                nc.close()
            except Exception as e:
                sys.stderr.write('Exception in ncutil.py diff(): {}\n'.format(e))

def install_nc(ifname, odir, installed=None):

    """Installs a netCDF file into odir.  Follows a convention for
    dependencies in the netCDF file:

        Attributes on the variable 'file' list files related to this
        file (absoulte paths).  Parallel attributes on the variable
        'install_paths' list the relative directory each file should
        be installed in.

    For example:

        int files ;
                files:source = "/home2/rpfische/modele_input/origin/GIC.144X90.DEC01.1.ext_1.nc" ;
                files:elev_mask = "/home2/rpfische/f15/modelE/init_cond/ice_sheet_ec/elev_mask.nc" ;
                files:icebin_in = "/home2/rpfische/f15/modelE/init_cond/ice_sheet_ec/icebin_in.nc" ;
        int install_paths ;
                install_paths:elev_mask = "landice" ;
                install_paths:icebin_in = "landice" ;

    Files without an install_path won't get installed."""

    if installed is None:
        installed = dict()    # ifname --> ofname

    # Make sure destination exists
    try:
        os.makedirs(odir)
    except:
        pass

    # Copy the netCDF file to the destination
    _,ifleaf = os.path.split(ifname)
    ofname = os.path.join(odir, ifleaf)
    print('Installing {} ->\n    {}'.format(ifname, ofname))
    shutil.copyfile(ifname, ofname)

    # Install dependencies of this file
    nc = None
    try:

        try:
            nc = netCDF4.Dataset(ofname, 'a')
        except RuntimeError:
            return

        try:
            files = nc.variables['files']
            install_paths = nc.variables['install_paths']
        except KeyError:
            # Files doesn't follow the conventions
            return

        # Iterate through attributes
        for label, relpath in install_paths.__dict__.items():
            child_ifname = getattr(files, label)
            _,child_leaf = os.path.split(child_ifname)
            child_odir = os.path.abspath(os.path.join(odir, relpath))
            child_ofname = os.path.join(child_odir, child_leaf)

            install_nc(child_ifname, child_odir)
            setattr(files, label, child_ofname)

    finally:
        if nc is not None:
            nc.close()

# -----------------------------------------------------------
@contextlib.contextmanager
def open(file, *args):
    """Context manager that either:
    a) If file is a a netCDF4 handle, just returns it
    b) Otherwise, opens netCDF4.Dataset(file, 'r')
    """

    if isinstance(file, netCDF4.Dataset):
        yield file
    else:
        with netCDF4.Dataset(file, *args) as nc:
            yield nc
# ====================================================
class NSVar(gicollections.MutableNamedTuple):
    __slots__ = ('dtype', 'dims', 'attrs')

def _var_schema(ncvar):
    # ncvar.dimensions is just a list of dimension NAMES
    return NSVar(ncvar.dtype, ncvar.dimensions,
        {name: ncvar.getncattr(name) for name in ncvar.ncattrs()})

class Schema:
    """Represents the schema of a NetCDF file (or a group in a NetCDF file).
    Can be used for custom copies of NetCDF file as follows:

        with netCDF4.Dataset('x.nc') as ncin:
            ncs = ncutil.Schema(ncin)
            with netCDF4.Dataset('y.nc', 'w') as ncout:
               ...modify schema here to control vars created...
                ncs.create(ncout, var_kwargs={'zlib': True})
               ...modify schema here to control vars copied...
                ncs.copy(ncin, ncout)
    """

    def __init__(self, ncin):
        """Returns the schema for a full NetCDF file (or a group therein)"""
        self.dims = {name: (None if ncdim.isunlimited() else len(ncdim)) \
            for name,ncdim in ncin.dimensions.items()}
        self.vars = {name: _var_schema(ncvar) for name,ncvar in ncin.variables.items()}
        self.attrs = {name: ncin.getncattr(name) for name in ncin.ncattrs()}
        self.groups = {name: Schema(val) for name,val in ncin.groups.items()}

    def create(self, ncout, var_kwargs=dict(zlib=True)):
        """Creates this schema in a new NetCDF file"""

        # Create dimensions
        for name,val in self.dims.items():
            ncout.createDimension(name, val)

        # Create variables
        for name,nsv in self.vars.items():
            ncv = ncout.createVariable(name, nsv.dtype, nsv.dims, **var_kwargs)
            for key,val in nsv.attrs.items():
                ncv.setncattr(key, val)

        # Create attributes
        for key,val in self.attrs.items():
            ncout.setncattr(key, val)

        # Create groups
        for key,nsg in self.groups.items():
            ncg = ncout.createGroup(key)
            nsg.create(ncg, var_kwargs=var_kwargs)

    def copy(self, ncin, ncout):
        """Copies this schema from ncin to ncout.
        create() must have already been run."""

        for vname in self.vars.keys():
            ncout.variables[vname][:] = ncin.variables[vname][:]

        # Copy groups
        for name,schema1 in self.groups.items():
            ncin1 = ncin.groups[name]
            ncout1 = ncout.groups[name]
            schema1.copy(ncin1, ncout1)
