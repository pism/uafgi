import os,subprocess
import netCDF4
from uafgi import make,ncutil
from uafgi.make import ncmake
import re



def fixup_for_pism(ipath, opath, tdir):
    """Fixup bedmachine file for use as PISM input file
    ipath:
        Name of BedMachine file
    odir:
        Place to put output BedMachine file
    tdir:
        Temporary directory
    """

    tmp1 = make.opath(ipath, tdir, '_fixup_pism1.nc')

    # Reverse the direction of the Y coordinate
    cmd = ['ncpdq', '-O', '-a', '-y', ifname, tmp1]
    subprocess.run(cmd, check=True)

    # Compress
    ncutil.compress(tmp1, ofname)


def replace_thk(bedmachine_file0, bedmachine_file1, thk):
    """Copies bedmachine_file0 to bedmachine_file1, using thk in place of original 'thickness'
    bedmachien_file0:
        Name of original BedMachine file
    bedmachine_file1:
        Name of output BedMachine file
    thk:
        Replacement thickness field"""

    with netCDF4.Dataset(bedmachine_file0, 'r') as nc0:
        with netCDF4.Dataset(bedmachine_file1, 'w') as ncout:
            cnc = ncutil.copy_nc(nc0, ncout)
            vars = list(nc0.variables.keys())
            cnc.define_vars(vars)
            for var in vars:
                if var not in {'thickness'}:
                    cnc.copy_var(var)
            ncout.variables['thickness'][:] = thk


class IceRemover(object):
    """Cuts local bedmachine files off at a terminus line, removing all ice downstream..."""

    def __init__(self, bedmachine_file):
        """bedmachine-file: Local extract from global BedMachine"""
        self.bedmachine_file = bedmachine_file
        with netCDF4.Dataset(self.bedmachine_file) as nc:

            bounding_xx = nc.variables['x'][:]
            bounding_yy = nc.variables['y'][:]

            # Determine Polygon of bounding box (xy coordinates; cell centers is OK)
            bb = (
                (bounding_xx[0],bounding_yy[0]), (bounding_xx[-1],bounding_yy[0]),
                (bounding_xx[-1],bounding_yy[-1]), (bounding_xx[0],bounding_yy[-1]))

            self.bounding_box = shapely.geometry.Polygon(bb)

            # ----- Determine regline: line going from end of 
            #dx = bounding_xx[-1] - bounding_xx[0]
            #dy = bounding_yy[-1] - bounding_yy[0]

            # Shift to cell edges (not cell centers)
            self.x0 = 2*bounding_xx[0] - bounding_xx[-1]    # x0-dx
            self.x1 = 2*bounding_xx[-1] - bounding_xx[0]    # x1+dx

            # ------- Read original thickness and bed
            self.thk = nc.variables['thickness'][:]
            self.bed = nc.variables['bed'][:]


    def get_thk(self, trace0):
        """Yields an ice thickness field that's been cut off at the terminus trace0
        trace0: (gline_xx,gline_yy)
            output of jsonutil.iter_traces()
        """

        # --------- Cut off ice at trace0
        # Convert raw trace0 to LineString gline0
        gline0 = shapely.geometry.LineString([
            (trace0[0][i], trace0[1][i]) for i in range(len(trace0[0]))])

        # Get least squares fit through the points
#        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(gline0[0], gline0[1])
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(trace0[0], trace0[1])

        regline = shapely.geometry.LineString((
            (self.x0, slope*self.x0 + intercept),
            (self.x1, slope*self.x1 + intercept)))


        # -------------- Intersect bounding box and lsqr fit to terminus
        intersection = self.bounding_box.intersection(regline)
        print('intersection ',list(intersection.coords))
        print(intersection.wkt)

        # -------------- Extend gline LineString with our intersection points
        intersection_ep = intersection.boundary
        gline_ep = gline0.boundary
        # Make sure intersection[0] is closets to gline[0]
        if intersection_ep[0].distance(gline_ep[0]) > intersection_ep[0].distance(gline_ep[1]):
            intersection_ep = (intersection_ep[1],intersection_ep[0])

        # Extend gline
        #print(list(intersection_ep[0].coords))
        print(intersection_ep[0].coords[0])
        glinex = shapely.geometry.LineString(
            [intersection_ep[0].coords[0]] + list(gline0.coords) + [intersection_ep[1].coords[0]])

        # Split our bounding_box polygon based on glinex
        # https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango
        merged = shapely.ops.linemerge([self.bounding_box.boundary, glinex])
        borders = shapely.ops.unary_union(merged)
        polygons = list(shapely.ops.polygonize(borders))

        with ioutil.tmp_dir() as tmp:

            # https://gis.stackexchange.com/questions/52705/how-to-write-shapely-geometries-to-shapefiles
        #    for i,poly in enumerate(polygons):
            i,poly = (0,polygons[0])
            if True:

                # Now convert it to a shapefile with OGR    
                driver = ogr.GetDriverByName('Esri Shapefile')
                poly_fname = os.path.join(tmp, 'poly{}.shp'.format(i))
                print('poly_fname ',poly_fname)
                ds = driver.CreateDataSource(poly_fname)
                layer = ds.CreateLayer('', None, ogr.wkbPolygon)

                # Add one attribute
                layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
                defn = layer.GetLayerDefn()

                ## If there are multiple geometries, put the "for" loop here

                # Create a new feature (attribute and geometry)
                feat = ogr.Feature(defn)
                feat.SetField('id', 123)

                # Make a geometry, from Shapely object
                geom = ogr.CreateGeometryFromWkb(poly.wkb)
                feat.SetGeometry(geom)

                layer.CreateFeature(feat)
                feat = geom = None  # destroy these

                # Save and close everything
                ds = layer = feat = geom = None

            # Mask out based on that polygon
            bed_masked_fname = os.path.join(tmp, 'bed_masked.nc')
        #    bed_masked_fname = 'x.nc'
            cmd =  ('gdalwarp', '-cutline', poly_fname, 'NETCDF:{}:bed'.format(self.bedmachine_file), bed_masked_fname)
            subprocess.run(cmd)

            # Read bed_maksed
            with netCDF4.Dataset(bed_masked_fname) as nc:
                bmask = nc.variables['Band1'][:].mask

        # Set bmask to the "downstream" side of the grounding line
        bmask_false = np.logical_not(bmask)
        if np.sum(np.sum(self.thk[bmask]==0)) < np.sum(np.sum(self.thk[bmask_false]==0)):
            bmask = bmask_false

        # Remove downstream ice
        thk = np.zeros(self.thk.shape)
        thk[:] = self.thk[:]
        thk[np.logical_and(bmask, self.bed<-100)] = 0

#        thk *= 0.5

        return thk

        ## Store it...
        #with netCDF4.Dataset(bedmachine_file, 'r') as nc0:
        #    with netCDF4.Dataset('x.nc', 'w') as ncout:
        #        cnc = ncutil.copy_nc(nc0, ncout)
        #        vars = list(nc0.variables.keys())
        #        cnc.define_vars(vars)
        #        for var in vars:
        #            if var not in {'thickness'}:
        #                cnc.copy_var(var)
        #        ncout.variables['thickness'][:] = thk





