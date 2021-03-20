import pandas as pd
import uafgi.data
from uafgi import pdutil,functional
import os,csv
import numpy as np

@functional.memoize
def read(map_wkt):
    # w21: Reads the dataset:
    # data/GreenlandGlacierStats/
    # 
    # From the paper:
    #
    # Ocean forcing drives glacier retreat in Greenland
    # Copyright © 2021
    #   Michael Wood1,2*, Eric Rignot1,2, Ian Fenty2, Lu An1, Anders Bjørk3,
    #   Michiel van den Broeke4, 11251 Cilan Cai , Emily Kane , Dimitris
    #   Menemenlis , Romain Millan , Mathieu Morlighem , Jeremie
    #   Mouginot1,5, Brice Noël4, Bernd Scheuchl1, Isabella Velicogna1,2,
    #   Josh K. Willis2, Hong Zhang2

    ddir = uafgi.data.join('GreenlandGlacierStats')

    keepcols = [
        ('Glacier (Popular Name)', 'popular_name'),
        ('Glacier (Greenlandic Name)', 'greenlandic_name'),
        ('Coast', 'coast'),
        ('Categorization', 'category'),
        ('Qr (km)', ('Qr','km', 'float')),
        ('Qf (km)', ('Qf', 'km', 'float')),
        ('Qm (km)', ('Qm', 'km', 'float')),
        ('Qs (km)', ('Qs', 'km', 'float')),
        ('Qc (Inferred) (km)', ('Qc_inferred', 'km', 'float')),
        ('qm (m/d)', ('qm', 'm d-1', 'float')),
        ('qf (m/d)', ('qf', 'm d-1', 'float')),
        ('qc (m/d)', ('qc', 'm d-1', 'float')),
        ('Mean Depth (m)', ('mean_depth', 'm', 'float')),
        ('Min Depth (m)', ('min_depth', 'm', 'float')),
        ('Quality', 'quality_str'),
        ('1992-2017 (Grounded, km2)', ('area_grounded_1992_2017', 'km^2', 'float')),
        ('1992-1997 (Grounded, km2)', ('area_grounded_1992_1997', 'km^2', 'float')),
        ('1998-2007 (Grounded, km2)', ('area_grounded_1998_2007', 'km^2', 'float')),
        ('2008-2017 (Grounded, km2)', ('area_grounded_2008_2017', 'km^2', 'float')),
        ('Mean Fjord Width (km)', ('mean_fjord_width', 'km', 'float')),
        ('1992-2017 (Grounded, km)', ('length_grounded_1992_2017', 'km', 'float')),
        ('1992-1997 (Grounded, km)', ('length_grounded_1992_1997', 'km', 'float')),
        ('1998-2007 (Grounded, km)', ('length_grounded_1998_2007', 'km', 'float')),
        ('2008-2017 (Grounded, km)', ('length_grounded_2008_2017', 'km', 'float')),
        ('Ocean Model Sample Area (See Figure 1)', 'ocean_model_sample_area'),
        ('1992-2017 (degrees C)', ('mean_TF_1992-2017', 'degC', 'float')),
        ('1992-1997 (degrees C)', ('mean_TF_1992-1997', 'degC', 'float')),
        ('1998-2007 (degrees C)', ('mean_TF_1998-2007', 'degC', 'float')),
        ('2008-2017 (degrees C)', ('mean_TF_2008-2017', 'degC', 'float')),
        ('1992-2017 (10^6 m^3/day)', ('subglacial_discharge_1992_2017', 'hm^3 d-1', 'float')),
        ('1992-1997 (10^6 m^3/day)', ('subglacial_discharge_1992_1997', 'hm^3 d-1', 'float')),
        ('1998-2007 (10^6 m^3/day)', ('subglacial_discharge_1998_2007', 'hm^3 d-1', 'float')),
        ('2008-2017 (10^6 m^3/day)', ('subglacial_discharge_2008_2017', 'hm^3 d-1', 'float')),
        ('Mean Cross-sectional Area (km^2)', ('mean_xsection_area', 'km^2', 'float')),
        ('1992-2017 (m/d)', ('mean_undercutting_1992_2017', 'm d-1', 'float')),
        ('1992-1997 (m/d)', ('mean_undercutting_1992_1997', 'm d-1', 'float')),
        ('1998-2007 (m/d)', ('mean_undercutting_1998_2007', 'm d-1', 'float')),
        ('2008-2017 (m/d)', ('mean_undercutting_2008_2017', 'm d-1', 'float')),
        ('Mean Undercutting Rate Uncertainty (%)', ('mean_undercutting_uncertainty', '%', 'float')),
        ('Flux Basin Mouginot et al 2019 (1)', 'flux_basin_mouginot_2019'),
        ('Mean Discharge (Gt/yr, 1992-2017)', ('mean_discharge', 'Gt a-1', 'float')),
        ('Mean Mass Balance (Gt/yr, 1992-2017)', ('mean_mass_balance', 'Gt a-1', 'float')),
        ('Reference SMB 1961-1990 (Gt/yr, 1961-1990)', ('reference_smb_1961_1990', 'Gt a-1', 'float')),
    ]


    dfs = list()
    for iroot in ('CW', 'SW', 'SE', 'CE', 'NE', 'N', 'NW'):
        ifname = os.path.join(ddir, '{}.csv'.format(iroot))
        # https://stackoverflow.com/questions/4869189/how-to-transpose-a-dataset-in-a-csv-file
        rows = list(zip(*csv.reader(open(ifname, "r"))))

        # Split up file
        columns = ['{} {}'.format(a,b).replace(':','').strip() for a,b in zip(rows[0], rows[1])]
        data = rows[2:]

        # Create dataframe
        df0 = pd.DataFrame(data, columns = columns)

        col_units = dict()
        df1_cols = dict()
        for col0,xx in keepcols:
            # Interpret convention for keepcols array
            if isinstance(xx,tuple):
                col1,units,dtype = xx
                col1 = col1
                col_units['w21_' + col1] = units
            else:
                col1 = xx
                units = None
                dtype = 'object'

            # Fish out old column
            col = df0[col0]
            if dtype == 'object':
                col.replace('N/A', '', inplace=True)
                col.replace('--', np.nan, inplace=True)
            elif dtype == 'float':
                col.replace('--', np.nan, inplace=True)
                col.replace('#DIV/0!', np.nan, inplace=True)
                col.replace('', np.nan, inplace=True)

            # Construct new set of columns
            df1_cols[col1] = col.astype(dtype)#.rename(col1)

        df = pd.DataFrame(df1_cols)

        # Remove wrong rows
        df = df[~df.popular_name.str.contains('Total')]
        df = df[~df.popular_name.str.contains('Mean')]

        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)


    #df.columns = ['w21_' + str(x) for x in df.columns]


    return pdutil.ext_df(df, map_wkt, add_prefix='w21_', units=col_units,
        keycols=['popular_name', 'flux_basin_mouginot_2019'],
        namecols=['popular_name', 'greenlandic_name'])

