from uafgi import cartopyutil

# WKT Strings for commonly used projections

# Standard Greenland Stereographic Projection
nsidc_ps_north = "PROJCS[\"WGS 84 / NSIDC Sea Ice Polar Stereographic North\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Polar_Stereographic\"],PARAMETER[\"latitude_of_origin\",70],PARAMETER[\"central_meridian\",-45],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"Easting\",SOUTH],AXIS[\"Northing\",SOUTH],AUTHORITY[\"EPSG\",\"3413\"]]"

wgs84 = epsg4326 = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'

greenland_extent = [-700000.0, 900000.0, -3500000.0, -500000.0]

def greenland_map(add_axes_fn, axes_bounds):
    """To get a map of Greenland:"""
    map_wkt = nsidc_ps_north
    crs = cartopyutil.crs(map_wkt)
    ax = add_axes_fn(axes_bounds, projection=crs)
    ax.set_extent(greenland_extent, crs=crs)
    return ax,map_wkt,crs

