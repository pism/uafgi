# ---------------------------------------------------------
def extend_raster(ras)
    """Extends a raster beyond its original bounds through successive Gaussian blurring.
    """

    ras_final = ras.copy()
    sigma = .25
    while sigma < 8:
        # Blur the image
        rasx = filterutil.nanfilter(ras, sigma, truncate=2.0)

        # Find the gridcells we will update
        print('xxxxxxxxx ', type(ras_final))
        mask_inx = np.logical_and(
                slope_mask_in,
                np.logical_and(
                    np.isnan(ras_final),
                    np.logical_not(np.isnan(rasx))))

        # Update those gridcells
        ras_final[mask_inx] = rasx[mask_inx]

        # Write output
         ofname = debug_dir / f'ras_{sigma:02.1f}.tif'
         gdalutil.write_raster(ofname, gridA, ras_final, -1000., type=gdal.GDT_Float32)

        # Iterate
        sigma *= 2

    ras = ras_final
    ras[ras<0] = 0
#    gdalutil.write_raster(debug_dir / 'ras.tif', gridA, ras, acsnowA_nd, type=gdal.GDT_Float32)

