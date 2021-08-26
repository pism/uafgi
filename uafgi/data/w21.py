import pandas as pd
import uafgi.data
from uafgi import pdutil,functional,cfutil,shputil
import os,csv
import numpy as np
import netCDF4
import scipy.interpolate
import itertools
import datetime
import scipy.integrate
import shapely

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


@functional.memoize
def raw_termini(map_wkt):
    df=pd.DataFrame(shputil.read(
        uafgi.data.join('wood2021', 'Greenland_Glacier_Ice_Front_Positions.shp'),
        read_shapes=True, wkt=map_wkt))
    df = df.rename(columns={'_shape':'terminus'}) \
        .drop(['_shape0'], axis=1)
    return df


def glacier_key(map_wkt):
    """Returns a mapping between Glacier and glacier_number;
    (two kinds of keys in Wood et al 2021)"""

    df = raw_termini(map_wkt)

    # Determine Wood 21 glacier_number ID by the ordering in the original file
    gndf = df[['Glacier']]
    gndf = gndf.drop_duplicates().reset_index(drop=True)
    gndf['glacier_number'] = gndf.index + 1

    return gndf

def read_termini(map_wkt):
    df = raw_termini(map_wkt)

    # Add a date column
#    print(df.columns)
    df['date'] = df[['Year', 'Month', 'Day']].apply(
        lambda x: datetime.datetime(*x), axis=1)

    gndf = glacier_key(map_wkt)


    # Join main df with glacier numbers
    df = pdutil.merge_nodups(df, gndf, on='Glacier', how='left')

    w21t = pdutil.ext_df(df, map_wkt,
        add_prefix='w21t_',
        keycols=['Glacier', 'Year', 'Day_of_Yea'])
    return w21t

def termini_by_glacier(w21t):
    """Collects rows from original read_termini() DataFrame by Glacier Name.
    Breaks the terminus lines apart into multiple points.
    Analogous to ns642.by_glacier_id()
    type: 'points'|'termini'
        points: Produce column with all points invovled in terminus LineStrings
        termini: Produce the entire terminus LineStrings, not broken up as points
    """

    # Create column with value [(date, terminus), ...]
    df2 = pdutil.group_and_tuplelist(w21t.df, ['w21t_Glacier'],
            [ ('w21t_date_termini', ['w21t_date', 'w21t_terminus']) ])
    xdf = w21t.replace(df=df2, keycols=['w21t_Glacier'])

    # Add w21t_glacier_number column
    gndf = glacier_key(w21t.map_wkt) 
    xdf.df = pdutil.merge_nodups(xdf.df, gndf, left_on='w21t_Glacier', right_on='Glacier').drop('Glacier', axis=1).rename(columns={'glacier_number':'w21t_glacier_number'})

    xdf.prefix = 'w21_'    # We're back to one row per glacier

    return xdf



# ========================================================


_var_specs = {

    ('ice_advection', 'rate') : ('ice_advection', 'rate', 'advection_rate'),
    ('ice_advection', 'cumulative') : ('ice_advection', 'cumulative_anomaly', 'cumulative_advection_anomaly'),
    ('ice_front_retreat', 'cumulative') : ('ice_front_retreat', 'discrete', 'retreat'),
#    ('ice_front_retreat', 'cumulative') : ('ice_front_retreat', 'smoothed', 'smoothed_retreat'),

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
#    print('year0 year1',year0,year1)



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
    
# https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def dec_to_datetime(decyr):
    year = int(decyr)
    rem = decyr - year
    base = datetime.datetime(year,1,1)
    return base + datetime.timedelta(seconds= (base.replace(year=base.year + 1) - base).total_seconds() * rem)

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
#                print(qname, qtype, 'value')
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
    times2 = np.linspace(year0, year1, int(.5+1+(year1-year0)/1.))
    # Jul 1 2010 -- Jun 30 2011 ==> label '2011'
    low_times2 = times2 - .5
    high_times2 = times2 + .5


    # Splinify each varaible
    cols = {'time': times2}
    for qname,qtype,_ in qnames:
        row = data[qname]

        # See if the quantity is already resampled to annual
        times = row['time']
        if times[1] - times[0] == 1.0:
            # Expand list of times, with NaN where no data.
            df2 = pd.DataFrame.from_dict({'time': times2})
            df = pd.DataFrame.from_dict({'time': row['time'], 'value': row['value']})
            df = pd.merge(df2, df, on='time', how='left')
            values2 = df['value']

            # Difference by year, if needed
            if qtype == 'cumulative':
                values2 = np.insert(np.diff(values2), 0, np.nan, axis=0)

        else:
            # Eliminate duplicate time values
            df = pd.DataFrame.from_dict({'time': row['time'], 'value': row['value']})
            df = df.groupby(['time']).mean()
            time = df.index.to_list()
            value = df['value'].to_list()

            # Spline interpolate existing data
#            F = scipy.interpolate.UnivariateSpline(row['time'], row['value'], k=1, ext=3)
            F = scipy.interpolate.interp1d(time, value,
                kind='cubic', bounds_error=False, fill_value=np.nan)


            if qtype == 'cumulative':
                # Rate is difference between end and beginning of each year
                values2 = F(high_times2) - F(low_times2)
            else:
                # Rate is integral of Spline over each year (divided by 1 year)
#                values2 = np.array([F.integral(lt, ht) for lt,ht in zip(low_times2, high_times2)])
                values2 = np.array([scipy.integrate.quad(F, lt, ht)[0] for lt,ht in zip(low_times2, high_times2)])
#                vals = list()
#                for lt,ht in zip(low_times2, high_times2):
#                    y,_,_,_,_ = 
#                print(values2)


        # Add to dataframe we're constructing
#        print(qname, values2.shape)
        cols[qname] = values2


    # Turn into a single dataframe
    df = pd.DataFrame.from_dict(cols)
    df = df.set_index('time')

    # Calving is the residual of advection, frontal retreat, front
    # undercutting and thinning-induced retreat
#    df['calving'] = df.sum(axis=1)
    df['calving'] = df.ice_front_retreat - df.ice_advection - df.ice_front_undercutting - df.thinning_induced_retreat

    return df

# ------------------------------------------------------------------
def velfile_df():
    """Produces a dataframe of velocity files"""
