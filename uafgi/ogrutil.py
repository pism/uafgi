from osgeo import ogr,osr

def reproject(src_ds, dst_srs, dst_ds):
    """Reprojects an entire OGR dataset.

    src_ds:
        Source dataset, eg:
            driver = ogr.GetDriverByName('GeoJSON')
            src_ds = driver.Open(polygon_file)

    dst_srs: osr.SpatialReference
        Destination SRS to convert to

    dst_ds:
        Destination dataset, eg one of:
           dst_ds = ogr.GetDriverByName('MEM').CreateDataSource('')
           dst_ds = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource('x.shp')
    """

    src_lyr = src_ds.GetLayer()   # Put layer number or name in her
    src_srs = src_lyr.GetSpatialRef()
    print('xxxxxxxxxxx ',type(src_srs), type(dst_srs))
    coordTrans = osr.CoordinateTransformation(src_srs, dst_srs)

    dst_lyr = dst_ds.CreateLayer(src_lyr.GetName(), srs=dst_srs)
    dst_lyr_def = dst_lyr.GetLayerDefn()

    # Copy fields from source to destination
    src_feature = src_lyr.GetFeature(0)
    for i in range(src_feature.GetFieldCount()):
        dst_lyr.CreateField(src_feature.GetFieldDefnRef(i))

    while True:
        in_feature = src_lyr.GetNextFeature()
        if in_feature is None:
            break

        # Reproject the feature
        geom = in_feature.GetGeometryRef()
        geom.Transform(coordTrans)
        dst_feature = ogr.Feature(dst_lyr_def)
        dst_feature.SetGeometry(geom)

        # Copy the fields
        for i in range(0, dst_lyr_def.GetFieldCount()):
            dst_feature.SetField(dst_lyr_def.GetFieldDefn(i).GetNameRef(), in_feature.GetField(i))

        # Finish
        dst_lyr.CreateFeature(dst_feature)
        dst_feature = None

    return dst_ds
