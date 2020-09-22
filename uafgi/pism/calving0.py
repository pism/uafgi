#!/usr/bin/env python

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

import PISM
from uafgi import argutil

def create_grid(ctx, filename, variable_name):
    """Create a shallow (2 vertical levels) computational grid by
    extracting grid information from a NetCDF variable.

    """
    return PISM.IceGrid.FromFile(ctx, filename, [variable_name], PISM.CELL_CENTER)

def init_geometry(grid, filename, min_ice_thickness):
    """Allocate storage for ice thickness and bed elevation (and other
    geometry-related fields) and initialize them from a NetCDF file.

    """
    # Geometry is a struct containing a bunch of IceModelVec instances.
    # It automatically pre-fills all the attributes in the constructor of Geometry.
    geometry = PISM.Geometry(grid)

    # Read the first last record of ice thickness and bed
    # elevation, stopping if not found
    geometry.ice_thickness.regrid(filename, critical=True)
    geometry.bed_elevation.regrid(filename, critical=True)
    geometry.sea_level_elevation.set(0.0)

    geometry.ensure_consistency(min_ice_thickness)

    return geometry

def init_velocity(grid, filename, record_index=0):
    """Allocate storage for ice velocity and initialize it from a file."""
    # allocate storage for ice velocity
    ice_velocity = PISM.model.create2dVelocityVec(grid, "_ssa_bc", stencil_width=2)
    ice_velocity.read(filename, record_index)

    # These two calls set internal and "human-friendly" units. Data read from a file will be
    # converted into internal units.
    # Ignore "input", long_name, internal_units, human_units, std_name, index into vector
    ice_velocity.set_attrs("input", "x-component of ice velocity", "m / s", "m / year", "", 0)
    ice_velocity.set_attrs("input", "y-component of ice velocity", "m / s", "m / year", "", 1)
#    ice_velocity.set_attrs("input", "x-component of ice velocity", "m / s", "m / s", "", 0)
#    ice_velocity.set_attrs("input", "y-component of ice velocity", "m / s", "m / s", "", 1)

    return ice_velocity

class FrontEvolution(object):

    default_kwargs = dict(
        ice_softness=3.1689e-24, sigma_max=1e6, max_ice_speed=5e-4)

    def __init__(self, grid, **kwargs0):
        self.kwargs = argutil.select_kwargs(kwargs0, self.default_kwargs)

        self.grid = grid
        self.ctx = grid.ctx()
        self.config = grid.ctx().config()

        # Use the sub-grid parameterization of the front position.
        self.config.set_flag("geometry.part_grid.enabled", True)

        self.min_thickness = self.config.get_number("geometry.ice_free_thickness_standard")

        # Allocate storage for ice enthalpy.
        #
        # Has to be there because model expects as part of input, but its content doesn't matter.
        #
        # It has a dummy vertical dimension with as few vertical
        # levels as possible, it only has 2 levels.
        self.ice_enthalpy = PISM.IceModelVec3(grid, "enthalpy", PISM.WITH_GHOSTS, 2)
        self.ice_enthalpy.set(0.0)

        # The GeometryEvolution class requires both the "advective"
        # (sliding) velocity and the SIA (diffusive) flux. We set it
        # to zero here.
        self.sia_flux = PISM.IceModelVec2Stag(grid, "sia_flux", PISM.WITHOUT_GHOSTS)
        self.sia_flux.set(0.0)

        # Allocate storage for ice velocity.
        self.ice_velocity = PISM.model.create2dVelocityVec(grid, "_ssa_bc", stencil_width=2)

        # Create the "Dirichlet BC" mask. This mask specifies
        # locations where ice thickness is fixed, i.e. prescribed as a BC.
        self.bc_mask = PISM.IceModelVec2Int(grid, "bc_mask", PISM.WITH_GHOSTS)

        # Storage for the retreat rate. Later on we will add the melt rate to this field.
        self.retreat_rate = PISM.IceModelVec2S(grid, "total_retreat_rate", PISM.WITHOUT_GHOSTS)
        self.retreat_rate.set_attrs("output", "rate of ice front retreat", "m / s", "m / day", "", 0)

        self.calving_model = self.create_calving_model(grid, self.kwargs['ice_softness'], self.kwargs['sigma_max'])

        self.retreat_model = PISM.FrontRetreat(grid)

        self.advance_model = PISM.GeometryEvolution(grid)

        self.iceberg_remover = PISM.IcebergRemover(grid)

        self.iceberg_remover.init()

    def set_bc_mask(self, velocity, bc_mask):
        """Set the BC mask to prevent ice thickness changes in areas where ice
        velocity is missing.

        """
        huge_value = 1e6

        bc_mask.set(0.0)

        with PISM.vec.Access([velocity, bc_mask]):
            for i, j in self.grid.points():
                speed = velocity[i, j].magnitude()
                if not speed < huge_value:
                    bc_mask[i, j] = 1

    def __call__(self, geometry, ice_velocity, run_length, output=None):
        """Perform a number of steps of the mass continuity equation and the
        calving parameterization to evolve ice geometry.

        """
        day_length = 86400.0

#        if report_filename is not None:
#            output = PISM.util.prepare_output(report_filename, append_time=False)
#        else:
#            output = None

        # create a copy of the velocity field. This copy will be
        # modified to cap ice speeds.
        self.ice_velocity.copy_from(ice_velocity)

        # Set the BC mask: 1 (fix ice thickness) where ice velocity is
        # missing, 0 (evolve thickness) elsewhere.
        self.set_bc_mask(self.ice_velocity, self.bc_mask)

        # Cap ice speed at ~15.8 km/year (5e-4 m/s).
        self.cap_ice_speed(self.ice_velocity, valid_max=self.kwargs['max_ice_speed'])

        t = 0.0
        while t < run_length:
            # compute the calving rate
            self.calving_model.update(geometry.cell_type, geometry.ice_thickness,
                                      self.ice_velocity, self.ice_enthalpy)

            self.retreat_rate.copy_from(self.calving_model.calving_rate())

            # Adjust the retreat rate to make sure that it does not
            # contain any NaNs or very large values.
            self.replace_missing(self.retreat_rate)

            dt_advance = PISM.max_timestep_cfl_2d(geometry.ice_thickness,
                                                  geometry.cell_type,
                                                  self.ice_velocity).dt_max.value()

            dt_retreat = self.retreat_model.max_timestep(geometry.cell_type,
                                                         self.bc_mask, self.retreat_rate).value()

            dt = min(dt_advance, dt_retreat)

            if t + dt > run_length:
                dt = run_length - t

            print(f"{t/day_length:.2f}, dt = {dt:.2f} (s) or {dt/day_length:2.2f} days")

            self.advance_model.flow_step(
                geometry, dt, self.ice_velocity,
                self.sia_flux, self.bc_mask)
            self.advance_model.apply_flux_divergence(geometry)

            geometry.ensure_consistency(self.min_thickness)

            self.retreat_model.update_geometry(
                dt, geometry, self.bc_mask, self.retreat_rate,
                geometry.ice_area_specific_volume,
                geometry.ice_thickness)

            # re-compute the cell type mask using 0 as the threshold
            # to clean up *all* icebergs, no matter how thin
            geometry.ensure_consistency(0.0)

            # Remove "icebergs" (left-over patches of floating ice not
            # connected to grounded ice).
            self.iceberg_remover.update(self.bc_mask,
                                        geometry.cell_type, geometry.ice_thickness)

            geometry.ensure_consistency(self.min_thickness)


            if output:
                PISM.append_time(output, self.config, t)

                geometry.ice_thickness.write(output)
                geometry.cell_type.write(output)
                self.retreat_rate.write(output)
                self.advance_model.flux_divergence().write(output)

            t += dt

    def cap_ice_speed(self, array, valid_max):
        "Cap velocity at valid_max"
        huge_value = 1e6
        with PISM.vec.Access(array):
            for i, j in self.grid.points():
                speed = array[i, j].magnitude()
                if not speed < huge_value:
                    # replace missing values
                    array[i, j].u = 0.0
                    array[i, j].v = 0.0
                elif speed > valid_max:
                    # cap speeds exceeding valid_max
                    C = valid_max / speed
                    array[i, j].u *= C;
                    array[i, j].v *= C;

        array.update_ghosts()

    def replace_missing(self, array, valid_max=1e6):
        """Eliminate too large (usually NaN) values resulting from missing
        input (usually ice velocity) values.

        """
        with PISM.vec.Access(array):
            for i, j in self.grid.points():
                if not array[i, j] < valid_max:
                    array[i, j] = 0.0

        array.update_ghosts()

    def create_calving_model(self, grid, ice_softness, sigma_max):
        """Create and initialize the calving model using the isothermal Glen
        flow law with a given ice softness.

        sigma_max:
            von Mises calving stress threshold
            See: https://github.com/pism/pism/blob/44db29423af6bdab2b5c990d08793010b2476cc5/doc/sphinx/manual/modeling-choices/marine/calving.rst
        """
        # This is a way to set the ice softness (and therefore hardness)
        # We will have to make a decision about this, we are not modeling T profile of ice
        #
        # It makes sense to use an isothermal flow law: ice softness is a prescribed constant
        # and hardness is related to softness.
        self.config.set_number("flow_law.isothermal_Glen.ice_softness", ice_softness)

        # allocate the flow law
        flow_law_factory = PISM.FlowLawFactory("calving.vonmises_calving.",
                                               self.ctx.config(), self.ctx.enthalpy_converter())
        flow_law_factory.set_default("isothermal_glen")
        flow_law = flow_law_factory.create()

        # set the tuning parameter of the calving law
        self.config.set_number("calving.vonmises_calving.sigma_max", sigma_max)

        # allocate and initialize the calving model
        model = PISM.CalvingvonMisesCalving(grid, flow_law)
        model.init()

        return model


if __name__ == "__main__":

    velocity_file   = 'outputs/TSX_W69.10N_2008_2020_pism_filled.nc'
    bedmachine_file = 'outputs/BedMachineGreenland-2017-09-20_pism_W69.10N.nc'

    ctx = PISM.Context()

    # ignore ice that is less than min_thickness meters thick
    min_ice_thickness = 50.0

    ctx.config.set_number("geometry.ice_free_thickness_standard", min_ice_thickness)

    grid = create_grid(ctx.ctx, bedmachine_file, "thickness")

    geometry = init_geometry(grid, bedmachine_file, min_ice_thickness)

    ice_velocity = init_velocity(grid, velocity_file)

    # NB: here I use a low value of sigma_max to make it more
    # interesting.
    front_evolution = FrontEvolution(grid, sigma_max=1e5)

    run_length_days = 120

    output = PISM.util.prepare_output(report_filename, append_time=False)

    front_evolution(geometry, ice_velocity,
                        run_length=run_length_days * 86400,
                        report_filename="test.nc")
