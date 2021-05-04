import pandas as pd
import uafgi.data
from uafgi import pdutil,functional,cfutil
import os,csv
import numpy as np
import netCDF4
import scipy.interpolate
import itertools

category_descr = {
    'DW' : 'Terminating in deep warm water (DW) with the detected presence of AW (warm Atlantic waters)',
    'CR' : 'Calving Ridge, independent of the presence of AW (warm Atlantic waters)',
    'SC' : 'stand in shallow cold (SC) (<100-m depth) fjords with polar water',
    'FE' : 'Glacier with long (>10km) floating extension',
    'NC' : 'Non-categorized due to lack of bathymetry and ocean temperature data'
}


#To simplify the analysis of the relationship between glacier undercutting and the onset and progression of glacier retreat, we group the 226 glaciers into six categories. Four categories are based on the fjord/ice geometry and the detected presence of AW, pertaining to 135 glaciers that have sufficient measurements to characterize bathymetry and water properties: (i) 74 glaciers terminating in deep warm water (DW) with the detected presence of AW; 

#(ii) 27 glaciers that break into icebergs on shallow ridges [calving ridges (CR)], independent of the presence of AW; (iii) 24 glaciers that stand in shallow cold (SC) (<100-m depth) fjords with polar water; and (iv) 10 glaciers with long (>10 km) floating extensions (FE). We partition the 91 remaining glaciers into two additional categories: (v) four glaciers already in an SR in 1992–1997 and (vi) 87 noncategorized (NC) glaciers due to a lack of bathymetry and ocean temperature data. The glacier distribution follows geography and precipitation regime (Fig. 3). DW glaciers dominate in NW, CW, and SE, where precipitation, glacier speed, and rates of mass turnover are high. SC glaciers are common in SW and CE, where mass turnover is lower. FE glaciers are in the cold, dry N and NE, except for Jakobshavn Isbræ and Alison. NC glaciers prevail in SW and CE where measurements of ice thickness and bathymetry are few, but the glaciers are thinner and the fjords are shallower than on average.


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

    ddir = uafgi.data.join('wood2021')

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


    do_reverse = {'SE', 'CE', 'NE'}    # These glaciers are reversed in Wood's numbering scheme
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

        # Reverse rows for some segments
        if iroot in do_reverse:
            df = df.iloc[::-1]

        dfs.append(df)

    df = pd.concat(dfs).reset_index(drop=True)
    # Add the glacier_number column used in Wood et al 2021's data files
    # (This works because we've read and added things in exactly the right order)
    df['glacier_number'] = df.index + 1

    # Join to the files index by glacier number
    index = pd.read_pickle(uafgi.data.join('wood2021', 'data', 'index.df')) \
        [['data_fname', 'glacier_number']]
    df = pdutil.merge_nodups(df, index, how='left', on='glacier_number')


    return pdutil.ext_df(df, map_wkt, add_prefix='w21_', units=col_units,
        keycols=['popular_name', 'flux_basin_mouginot_2019'],
        namecols=['popular_name', 'greenlandic_name'])

# ========================================================


_var_specs = {

    ('ice_advection', 'rate') : ('ice_advection', 'rate', 'advection_rate'),
    ('ice_advection', 'cumulative') : ('ice_advection', 'cumulative_anomaly', 'cumulative_advection_anomaly'),
    ('ice_front_retreat', 'cumulative') : ('ice_front_retreat', 'discrete', 'retreat'),

    ('ice_front_undercutting', 'rate'): ('ice_front_undercutting', 'rate', 'undercutting_rate'),
    ('ice_front_undercutting','cumulative'): ('ice_front_undercutting', 'cumulative_anomaly', 'cumulative_undercutting_anomaly'),

    ('thinning_induced_retreat', 'cumulative'): ('thinning_induced_retreat', 'thinning_induced_retreat'),
}

_subvar_suffix = {
    'time': '_time',
    'value': '',
    'uncertainty': '_uncertainty',
}

def data_var(nc, qname, qtype, subvar):
    """
    qname:
        Name of the quantity being sought.
        Eg: 'ice_advection', 'ice_front_retreat',
            'ice_front_undercutting', 'thinning_induced_retreat'
    qtype:
        Whether we want instantaneous ('rate') or cumulative anomaly ('cumulative')
    subvar:
        Which NetCDF variable for the quantity timeseries we want.
        Eg: 'time', 'value', 'uncertainty'
    """
    spec = _var_specs[(qname, qtype)]
    group = nc
    for gname in spec[:-1]:
        group = group.groups[gname]
    vstem = spec[-1]    # Stem of variable names

    return group.variables[vstem + _subvar_suffix[subvar]]


def open_data(w21_data_fname):
    """Returns full pathname of a datafile.
    w21_data_fname:
        Name of the data file (from the w21 dataframe above)
    """
    ifname = uafgi.data.join('wood2021', 'data', w21_data_fname)
    return netCDF4.Dataset(ifname)

# ------------------------------------------------------------
def glacier_cumulative_df(data_fname):
    """Retrieves cumulative glacier data for one glacier, (sort of)
    duplicating plots in the Wood et al 2021 paper.

    Returns: pd.DataFrame
        Cumulative effect of processes causing advance/retreat of the terminus.
        Postive numbers always advance the terminus, negative for retreat.
        May be plotted with `df.plot()`
    Columns:
        year (index):
            Year of the data
        ice_advection: [km]
        ice_front_retreat: [km]
        ice_front_undercutting: [km]
        thinning_induced_retreat: [km]
        calving: [km]
            Calving, calucated as sum of the other columns
        calving_rate: [km a-1]
            Successive differences of calving
    """

    qnames = (
            ('ice_advection',1), ('ice_front_retreat',-1), ('ice_front_undercutting',-1),
            ('thinning_induced_retreat',-1))

    # Read the data
    data = dict()
    tt0 = 1000000
    tt1 = -1000000
    with open_data(data_fname) as nc:
        for qname,sign in qnames:
            time = data_var(nc, qname, 'cumulative', 'time')[:].data
            value = data_var(nc, qname, 'cumulative', 'value')[:].data
            tt0 = min(tt0, np.min(time))
            tt1 = max(tt1, np.max(time))


            data[qname] = {
                'qname': qname, 'w21_data_fname':data_fname, 'time':time,
                'value':value*sign,
            }

    year0 = np.floor(tt0)
    year1 = np.ceil(tt1)
    print('year0 year1',year0,year1)



    # Average by year
    for qname in ('ice_front_retreat',):
        row = data[qname]

        df = pd.DataFrame(columns=('time', 'value'), data={'time':row['time'], 'value':row['value']})
        df['iyear'] = (df['time'] - .5).apply(np.floor)
        df = df.groupby(['iyear']).mean()
        row['time'] = df.index.to_list()
        row['value'] = df.value.to_list()

    # Find range of years

    # Interpolate to 1x/year
    time2 = np.linspace(year0, year1, int(.5+1+(year1-year0)/1.))
    cols = {'time': time2}
    for qname,row in data.items():
        F = scipy.interpolate.interp1d(row['time'], row['value'], fill_value='extrapolate')
        year0 = 1990.
        year1 = 2018.
        cols[qname] = F(time2)

    # Turn into a single dataframe
    df = pd.DataFrame.from_dict(cols)
    df = df.set_index('time')

    # Calving is the residual of advection, frontal retreat, front
    # undercutting and thinning-induced retreat
    df['calving'] = df.sum(axis=1)

    return df
    
def glacier_rate_df(data_fname):
    """Retrieves cumulative glacier data for one glacier, (sort of)
    duplicating plots in the Wood et al 2021 paper.

    Returns: pd.DataFrame
        Cumulative effect of processes causing advance/retreat of the terminus.
        Postive numbers always advance the terminus, negative for retreat.
        May be plotted with `df.plot()`
    Columns:
        year (index):
            Year of the data
        ice_advection: [km]
        ice_front_retreat: [km]
        ice_front_undercutting: [km]
        thinning_induced_retreat: [km]
        calving: [km]
            Calving, calucated as sum of the other columns
        calving_rate: [km a-1]
            Successive differences of calving
    """

    qnames = (
            ('ice_advection','rate',1), ('ice_front_undercutting','rate',-1),
            ('ice_front_retreat','cumulative',-1), ('thinning_induced_retreat','cumulative',-1))

    # Read the data
    data = dict()
    tt0 = 1000000
    tt1 = -1000000

    with open_data(data_fname) as nc:
        # Read rate variables
        for qname,qtype,sign in qnames:

            # Read the raw variable and convert units
            time = data_var(nc, qname, qtype, 'time')[:].data
            nc_value = data_var(nc, qname, qtype, 'value')
            if qtype == 'rate':
                print(qname, qtype, 'value')
                value = cfutil.convert(nc_value[:].data, nc_value.units, 'km yr-1')
            else:
                value = cfutil.convert(nc_value[:].data, nc_value.units, 'km')

            tt0 = min(tt0, np.min(time))
            tt1 = max(tt1, np.max(time))

            data[qname] = {
                'qname': qname, 'w21_data_fname':data_fname, 'time':time,
                'value':value*sign,
            }

    # Timepoints to interpolate to, 1x/yr
    year0 = np.floor(tt0)
    year1 = np.ceil(tt1)
    year0 = 1995
    year1 = 2017
    times = np.linspace(year0, year1, int(.5+1+(year1-year0)/1.))


    # Splinify each varaible
    cols = {'time': times}
    for qname,qtype,_ in qnames:
        row = data[qname]

        # Get a Least Square Spline of the RATE
        t0 = row['time'][0]
        t1 = row['time'][-1]
        knots = row['time']
        #knots = knots[np.logical_and(knots > year0, knots < year1)]
        #knots = knots[1:-1]

        print('Splinify ',qname, len(row['time']), 2*len(knots))
        if len(knots) <= 2*len(times):
#            F = scipy.interpolate.interp1d(row['time'],row['value'])
            F = scipy.interpolate.UnivariateSpline(row['time'], row['value'], knots, k=3)
            values = F(times)
            if qtype == 'cumulative':
                F = F.derivative()
#            if qtype == 'cumulative':
#                values = np.insert(values.diff(), 0, np.nan, axis=0)
        else:
            print('LSQUSpline ',qname)
            F = scipy.interpolate.LSQUnivariateSpline(row['time'], row['value'], knots)
            if qtype == 'cumulative':
                F = F.derivative()
            values = F(times)

        # Add to dataframe we're constructing
        cols[qname] = values

    # Turn into a single dataframe
    df = pd.DataFrame.from_dict(cols)
    df = df.set_index('time')

    # Calving is the residual of advection, frontal retreat, front
    # undercutting and thinning-induced retreat
    df['calving'] = -df.sum(axis=1)

    return df
