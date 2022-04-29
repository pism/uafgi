import netCDF4
import PISM
import cf_units
from uafgi import cfutil,argutil
from uafgi.pism import calving0


class compute(object):

    """This script reads in data from a file and runs PISM's von Mises calving model.
    The calving rate is saved to "output.nc".
    The calving model has the following inputs
    - x and y components of the ice velocity
    - "cell type" mask (used to locate the calving front)
    The following three are used to compute the vertically-averaged ice hardness:
    - ice thickness
    - ice enthalpy
    - the flow law
    This script uses the isothermal flow law, so ice thickness is irrelevant but should be
    positive in the ice-covered area. The enthalpy value does not matter.
    """

    default_kwargs = dict(calving0.FrontEvolution.default_kwargs.items())
    default_kwargs['min_ice_thickness'] = 50.0

    def __init__(self, makefile, geometry_file, velocity_file, grid_fivar, output_file, **kwargs0):
        """Runs the PISM calving model on a timeseries of velocity fields.

        geometry_file: (BedMachine file)
            File containing the following variables typical of a PISM
            state file (in addition to the coordinate variables x and
            y and optional time t): Only the LAST time index will be
            read, and regridded to the grid from velocity_file.

            double thk(t, y, x) ;
                thk:units = "meters" ;
                thk:standard_name = "land_ice_thickness" ;
                thk:coordinates = "lon lat" ;
            float topg(t, y, x) ;
                topg:long_name = "bedrock topography" ;
                topg:units = "meter" ;
                topg:standard_name = "bedrock_altitude" ;
                topg:coordinates = "lon lat" ;

        velocity_file:
            File containing surface velocities:
            (in addition to the coordinate variables x and y and optional time t):
            All time indices will be read.
            Grid will be taken from this file.

            float u_ssa_bc(time, y, x) ;
                u_ssa_bc:units = "m / year" ;
                u_ssa_bc:coordinates = "lat lon" ;
                u_ssa_bc:Content = "Ice velocity in x direction" ;
                u_ssa_bc:Units = "  meter/year" ;
                u_ssa_bc:_FillValue = -9.e+33f ;
                u_ssa_bc:missing_value = -9.e+33f ;
            float v_ssa_bc(time, y, x) ;
                v_ssa_bc:units = "m / year" ;
                v_ssa_bc:coordinates = "lat lon" ;
                v_ssa_bc:Content = "Ice velocity in y direction" ;
                v_ssa_bc:Units = "  meter/year" ;
                v_ssa_bc:_FillValue = -9.e+33f ;
                v_ssa_bc:missing_value = -9.e+33f ;

        grid_fivar: (file, (variable, ...))
            File/variable pair from which to read the (2D) grid

        output_file:
            Name of file to be created, will contain variables:

            // Calving rate, spatially distributed
            double vonmises_calving_rate(time, y, x) ;
                vonmises_calving_rate:units = "m year-1" ;
                vonmises_calving_rate:long_name = "horizontal calving rate due to von Mises calving" ;
                vonmises_calving_rate:pism_intent = "diagnostic" ;

        min_ice_thickness: [m]
            ignore ice thinner than min_thickness
        sigma_max:
            ??? parameter inside PISM???
        """
        self.kwargs = argutil.select_kwargs(kwargs0, self.default_kwargs)

        self.geometry_file = geometry_file
        print('geometry_file = {}'.format(self.geometry_file))
        self.velocity_file = velocity_file
        print('velocity_file = {}'.format(self.velocity_file))
        self.grid_fivar = grid_fivar
        self.kwargs = argutil.select_kwargs(kwargs0, self.default_kwargs)

        self.rule = makefile.add(self.run,
            (geometry_file, velocity_file, grid_fivar[0]),
            (output_file,))

    def run(self):

        print('========== calving.compute.run()')
        print('velocity_file = {}'.format(self.velocity_file))
        print('geometry_file = {}'.format(self.geometry_file))

        #### I need to mimic this: Ross_combined.nc plus the script that made it
        # Script in the main PISM repo, it's in examples/ross/preprocess.py
        #self.geometry_file = "~/github/pism/pism/examples/ross/Ross_combined.nc"
        # self.geometry_file = "Ross_combined.nc"
        ctx = PISM.Context()
        # TODO: Shouldn't this go in calving0.init_geometry()?
        ctx.config.set_number("geometry.ice_free_thickness_standard", self.kwargs['min_ice_thickness'])

        grid = calving0.create_grid(ctx.ctx, self.geometry_file, "thickness")
        geometry = calving0.init_geometry(grid, self.geometry_file, self.kwargs['min_ice_thickness'])
        ice_velocity = calving0.init_velocity(grid, self.velocity_file)

        # NB: here I use a low value of sigma_max to make it more
        # interesting.
        front_evolution = calving0.FrontEvolution(grid, **self.kwargs)
#            ice_softness=self.kwargs['ice_softness'],
#            sigma_max=self.kwargs['sigma_max'])

# --- This was not in Constantin'es original script
        # Read the time variable
        print('vvvvelocity ',self.velocity_file)
        with netCDF4.Dataset(self.velocity_file, 'r') as nc:
            nctime = nc.variables['time']
            time_units = cf_units.Unit(nctime.units, nctime.calendar)
            sec_units = cfutil.replace_reftime_unit(time_units, 'seconds')
            nctime_bnds_d = nc.variables[nctime.bounds]
            time_d = nctime[:]    # Time in days
            timeattrs = dict(
                (name,nctime.getncattr(name))
                for name in nctime.ncattrs()) # All attrs on time var
            timeattrs['units'] = str(sec_units)    # PISM will write in seconds!
            time_bnds_d = nctime_bnds_d[:]
# -----------

        # Open the output file
        # The append_time=True argument of prepare_output determines if
        # after this call the file will contain zero (append_time=False)
        # or one (append_time=True) records.
        output = PISM.util.prepare_output(self.rule.outputs[0], append_time=False)

        try:
            # Add in all the time attributes
            for name,val in timeattrs.items():
                output.write_attribute('time', name, val)

            # Run the calving model for each different forcing through time
            for its in range(0,len(time_d)):
                tb_d = time_bnds_d[its,:]
                tb_s = time_units.convert(tb_d, sec_units)

                ice_velocity.read(self.velocity_file, its)   # 0 ==> first record of that file (if time-dependent)

                front_evolution(geometry, ice_velocity,
                    tb_s[0], tb_s[1],
                    output=output)

        finally:
            output.close()

            # Add dummy var to output_file; helps ncview
            with netCDF4.Dataset(self.rule.outputs[0], 'a') as nc:
                nc.createVariable('dummy', 'i', ('x',))



# To go from Numpy array into PETSc:
# https://github.com/pism/pism/blob/master/test/miscellaneous.py#L784-L788
#test/miscellaneous.py:784-788
#    with PISM.vec.Access(nocomm=[thk1]):
#        for (i, j) in grid.points():
#            F_x = (x[i] - x_min) / (x_max - x_min)
#            F_y = (y[j] - y_min) / (y_max - y_min)
#            thk1[i, j] = (F_x + F_y) / 2.0
