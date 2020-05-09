import netCDF4
import PISM
import cf_units
from uafgi import cfutil

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

    def __init__(self, makefile, geometry_file, velocity_file, grid_fivar, output_file, ice_softness=3.1689e-24):
        """Runs the PISM calving model on a timeseries of velocity fields.

        geometry_file:
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

        """
        self.geometry_file = geometry_file
        self.velocity_file = velocity_file
        self.grid_fivar = grid_fivar
        self.ice_softness = ice_softness
        self.rule = makefile.add(self.run,
            (geometry_file, velocity_file, grid_fivar[0]),
            (output_file,))

    def run(self):

        #### I need to mimic this: Ross_combined.nc plus the script that made it
        # Script in the main PISM repo, it's in examples/ross/preprocess.py
        #self.geometry_file = "~/github/pism/pism/examples/ross/Ross_combined.nc"
        # self.geometry_file = "Ross_combined.nc"
        ctx = PISM.Context()
        config = ctx.config

        # This is a way to set the ice softness (and therefore hardness)
        # We will have to make a decision about this, we are not modeling T profile of ice
        # It makes sense to use an isothermal flow law: ice softness is a prescribed constant
        # and hardness is related to softness.
        config.set_number("flow_law.isothermal_Glen.ice_softness", self.ice_softness)

        # get grid information from the variable "thk" in a file.
        # (Not a full-blown 3D grid, by getting from thk which is a 2D variable)
        # grid = PISM.IceGrid.FromFile(ctx.ctx, self.geometry_file, ["thk"], PISM.CELL_CENTER)
        grid = PISM.IceGrid.FromFile(ctx.ctx, self.grid_fivar[0], self.grid_fivar[1], PISM.CELL_CENTER)

        # allocate storage for ice enthalpy
        # Has to be there because model expects as part of input, but its content doesn't matter.
        # It has a dummy vertical dimension with as few vertical levels as possible,
        # it only has 2 levels.
        ice_enthalpy = PISM.IceModelVec3(grid, "enthalpy", PISM.WITH_GHOSTS, 2)
        ice_enthalpy.set(0.0)

        # allocate storage for ice velocity
        # 2V ===> Vectorfield, allocates 2 fields, stored interlaced in RAM, separately in files.
        ice_velocity = PISM.IceModelVec2V(grid, "_ssa_bc", PISM.WITH_GHOSTS, 2)

        # These two calls set internal and "human-friendly" units. Data read from a file will be
        # converted into internal units.
        # Ignore "input", long_name, internal_units, human_units, std_name, index into vector
#        ice_velocity.set_attrs("input", "x-component of ice velocity", "m / s", "m / year", "", 0)
#        ice_velocity.set_attrs("input", "y-component of ice velocity", "m / s", "m / year", "", 1)
        ice_velocity.set_attrs("input", "x-component of ice velocity", "m / s", "m / s", "", 0)
        ice_velocity.set_attrs("input", "y-component of ice velocity", "m / s", "m / s", "", 1)

        # Read the time variable
        with netCDF4.Dataset(self.velocity_file, 'r') as nc:
            nctime = nc.variables['time']
            time_units = cf_units.Unit(nctime.units, nctime.calendar)
            sec_units = cfutil.replace_reftime_unit(time_units, 'seconds')
            nctime_bnds_d = nc.variables[nctime.bounds]
            time_d = nctime[:]    # Time in days
            timeattrs = [(name,nctime.getncattr(name)) for name in nctime.ncattrs()] # All attrs on time var
            time_bnds_d = nctime_bnds_d[:]

        # Geometry is a struct containing a bunch of these IceModelVec instances.
        # It automatically pre-fills the constructor of Geometry, all the attributes.
        # It's easier to just do that, rather than allocating the handful of them we may need.
        # allocate storage for all geometry-related fields. This does more than we need (but it's
        # easy).
        geometry = PISM.Geometry(grid)

        # read the first (0th) record of ice thickness and bed elevation
        # TODO: Use regrid() instead of read() if we expect to be interpolating from another grid.
        geometry.ice_thickness.regrid(self.geometry_file)
        geometry.bed_elevation.regrid(self.geometry_file)
        geometry.sea_level_elevation.set(0.0)

        # Compute ice_free_thickness_standard based on what we just set above.
        # We're grabbing ice_free_thickness_standard from our config database and
        # using it to compute surface elevation and cell type.
        # Generally, think of ice_free_thickness_standard == 0
        # ensure consistency of geometry (computes surface elevation and cell type)
        geometry.ensure_consistency(config.get_number("geometry.ice_free_thickness_standard"))
        min_thickness = ctx.config.get_number("geometry.ice_free_thickness_standard")

        print('AA2')

        # allocate the flow law
        flow_law_factory = PISM.FlowLawFactory(
            "calving.vonmises_calving.", ctx.config, ctx.enthalpy_converter)
        flow_law_factory.set_default("isothermal_glen")
        flow_law = flow_law_factory.create()

        print('A31')
        bc_mask = PISM.IceModelVec2Int(grid, "bc_mask", PISM.WITH_GHOSTS)
        bc_mask.set(0.0)

        retreat_rate = PISM.IceModelVec2S(grid, "total_retreat_rate", PISM.WITHOUT_GHOSTS)
        retreat_rate.set_attrs("output", "rate of ice front retreat", "m / s", "m / s", "", 0)
        retreat_rate.set(0.0)

        melt_rate = PISM.IceModelVec2S(grid, "frontal_melt_rate", PISM.WITHOUT_GHOSTS)
        melt_rate.set(0.0)


        # Open the output file
        # The append_time=True argument of prepare_output determines if
        # after this call the file will contain zero (append_time=False)
        # or one (append_time=True) records.
        output = PISM.util.prepare_output(self.rule.outputs[0], append_time=False)

        try:
            # Add in all the time attributes
            for name,val in timeattrs:
                output.write_attribute('time', name, val)

            # allocate and initialize the calving model
            model = PISM.CalvingvonMisesCalving(grid, flow_law)
            model.init()

            front_retreat = PISM.FrontRetreat(grid)

            # Run the calving model for each timestep
            for its in range(0,len(time_d)):
                tb_d = time_bnds_d[its,:]
                tb_s = time_units.convert(tb_d, sec_units)

                ice_velocity.read(self.velocity_file, its)   # 0 ==> first record of that file (if time-dependent)


                print('********** Computing Calving for ({} - {})'.format(time_units.num2date(time_bnds_d[its,0]), time_units.num2date(time_bnds_d[its,1])))

                # Timestep through the time bounds (in seconds)
                t_s = tb_s[0]
                while t_s < tb_s[1]:
                    print('----- t={}'.format(sec_units.num2date(t_s)))

                    # compute the calving rate
                    model.update(
                        geometry.cell_type, geometry.ice_thickness,
                        ice_velocity, ice_enthalpy)

                    # combine calving rate with the frontal melt rate to produce the total retreat rate
                    retreat_rate.copy_from(model.calving_rate())
                    retreat_rate.add(1.0, melt_rate)
                

                    # compute the maximum allowed time step length
                    dtmax_s = front_retreat.max_timestep(
                        geometry.cell_type,
                        bc_mask, retreat_rate).value()
                    print('dtmax_s = {}s ({} days)'.format(dtmax_s, dtmax_s/86400))
                    dt_s = min(dtmax_s, tb_s[1] - t_s)

                    front_retreat.update_geometry(
                        dt_s, geometry, bc_mask, retreat_rate,
                        geometry.ice_area_specific_volume,
                        geometry.ice_thickness)
                    geometry.ensure_consistency(min_thickness)

                    t_s += dt_s

                    # save results (could be innner or outer loop)
                    PISM.append_time(output, "time", sec_units.convert(t_s, time_units))
#                    ice_velocity = 1.0
                    ice_velocity.write(output)

                    model.calving_rate().set_attrs(    # Redundant
                        "output", "vonmises_calving_rate", "m / s", "m / s", "", 0)
                    model.calving_rate().write(output)

                    retreat_rate.write(output)
                    geometry.ice_thickness.write(output)



                # Writes just a scalar: movement of the front in horizontal direction, by how much (s-1)
                # the front retreats.  Currently, how it's used in PISM, it's computed at ice-free locations
                # immediately adjacent to the calving front.  That's where it would be applied by
                # PISM's parameterization of sub-grid position of the calving front.
                # save to an output file
                # See pism.git/site-packages/pism/util.py
                #model.calving_rate().write(output)
    #            geometry.ice_thickness.write(output)    # One more thing to write
        finally:
            output.close()

                # this is a way to access the calving rate in a Python script without saving to a file
                # rate = model.calving_rate().numpy()

        # Add dummy var to output_file; helps ncview
        with netCDF4.Dataset(self.rule.outputs[0], 'a') as nc:
            nc.createVariable('dummy', 'i', ('x',))
