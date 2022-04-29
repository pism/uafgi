from osgeo import gdal

def read(fname, iband):
    """Reads a GeoTIFF file, returns the raster.
    fname:
        Name of GeoTIFF file toread
    iband: int (1-based)
        Band in the GeoTIFF file to read

    NOTE: Try inspecting your file with the command:
        gdalinfo myfile.tif
    """

    # open dataset
    ds = gdal.Open(fname)
    srcband = ds.GetRasterBand(iband)
    return srcband.ReadAsArray()

    for band in range(1,ds.RasterCount+1):
        srcband = ds.GetRasterBand(band)
        arr = srcband.ReadAsArray()
