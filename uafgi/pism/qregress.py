import os
import numpy as np
import pandas as pd
import netCDF4
import scipy.interpolate
import statsmodels.api
import statsmodels.formula.api
from uafgi.data import w21 as d_w21
from uafgi import pdutil
import copy

# -------------------------------------------------------------------
def fit_sigma_maxs(select, velterm_df):
    """Determine the sigma_max that fits the data for each
    velocity/terminus combo."""


    # Use only concurrent velocity / terminus measurements; and average within each year
    vtdf = copy.copy(velterm_df)
    vtdf['ivel_year'] = vtdf['vel_year'].apply(np.floor)
    vtdf['iterm_year'] = vtdf['term_year'].apply(np.floor)
    vtdf = vtdf[vtdf.ivel_year == vtdf.iterm_year]
    #vtdf = vtdf.groupby('ivel_year').mean().reset_index()
    vtdf = vtdf[['up_area', 'aflux', 'sflux', 'glacier_id', 'ivel_year']]
    vtdf = vtdf.rename(columns={'ivel_year':'year'})

#    rows = list()
    dfs = list()
    for _,selrow in select.df.iterrows():
        adv = vtdf[vtdf.glacier_id == selrow.w21t_glacier_number]
        
        df = d_w21.glacier_rate_df(selrow.w21_data_fname)
        df = df.reset_index()

        df = pdutil.merge_nodups(df, adv, left_on='time', right_on='year', how='left').drop('year',axis=1)
#        df = pdutil.merge_nodups(df, adv, left_on='time', right_on='term_year', how='left').drop('year',axis=1)

        #df['sflux1'] = df['sflux'] * df['ice_advection'] / df['aflux']

        df['sigma_max'] = (df['sflux'] * df['ice_advection']) / (df['aflux'] * -df['calving'])
        df = df.reset_index()
        df = df[['glacier_id', 'time', 'sigma_max']].dropna()
#        rows.append({'glacier_id': selrow.w21t_glacier_number, 'sigma_max_mean': df.sigma_max.mean(), 'sigma_max_std': df.sigma_max.std()})
        dfs.append(df)
#        break

    return pd.concat(dfs)
#    return pd.DataFrame(rows)
# -------------------------------------------------------------------
def _selcols(select, y0, y1):
    y_end = y1-1
    Qsg_name = f'w21_subglacial_discharge_{y0:04d}_{y_end:04d}'
    TF_name = f'w21_mean_TF_{y0:04d}-{y_end:04d}'
        
    df0 = select.df[[
        'w21t_Glacier', 'w21t_glacier_number', 'w21_coast', 'w21_mean_fjord_width',
        Qsg_name, TF_name]]
    df0 = df0.rename(columns={Qsg_name:'Qsg', TF_name:'TF'})
    df0['q4tf'] = df0['Qsg']**0.4 * df0['TF']
    df0['year0'] = float(y0)
    df0['year1'] = float(y1)
    return df0

def wood_q4tf(select):

    """Given our `select` dataset (d_stability), creates a dataframe with
    Qsd and TF by year range.  (This should really go in d_stability.py)

    Returns wdfs[] 2 dataframes:
        wdfs[0] = dataframe for glaciers in regions SE,SW,CE,CW,NE
        wdfs[1] = dataframe for glaciers in regions N,NW

    Returned columns: 
        w21t_Glacier
        w21t_glacier_number
        w21_coast
        w21_mean_fjord_width
        Qsg:
            Subglacial discharge
        TF: [degC]
            Thermal Forcing in fjord
        q4tf:
            Qsg^.04 * TF
        year0:
            First year for which q4tf is valid
        year1:
            Last year (+1) for which q4tf is valid
    """

    dfs = list()
    for y0,y1 in ((1992,1998), (1998,2008), (2008,2018)):
        df = _selcols(select, y0,y1)
        dfs.append(df)
    df = pd.concat(dfs)

    #df0 = _selcols(select, 1992, 1998)#'w21_subglacial_discharge_1998_2007', 'w21_mean_TF_1998-2007')
    #df1 = _selcols(select, 'w21_subglacial_discharge_2008_2017', 'w21_mean_TF_2008-2017')

    #df = pd.concat((df0,df1)).sort_values('w21t_Glacier')
    # See Estimating Greenland tidewater glacier retreat driven by submarine melting (Slater et al 2019) p. 2497
    wdfs = [df[df.w21_coast.isin(['SE','SW','CE','CW','NE'])],  df[df.w21_coast.isin(['N','NW'])]]
    return wdfs

# -------------------------------------------------------------------
def read_retreats(select):
    """Reads the per-glacier Wood data files to retrieve linear retreat.
    Returns dataframe:
        year:
            Time of measurement (can be franctional year)
        up_len:
            Number gets smaller as glacier retreats; can be negative.
            This column is de-meaned
        w21t_glacier_number:
    """

    # Read year,up_len from Wood et al data
    rdfs = list()

    year_bounds = [1992,]

    for _,selrow in select.df.iterrows():
        with netCDF4.Dataset(os.path.join('data', 'wood2021', 'data', selrow['w21_data_fname'])) as nc:
            grp = nc.groups['ice_front_retreat']['discrete']
            retreat_time = grp.variables['retreat_time'][:].astype(np.float64)
            retreat = grp.variables['retreat'][:]
            retreat -= np.mean(retreat)   # Retreat is compared to average; zero point doesn't really matter
            rdf = pd.DataFrame({'year':retreat_time, 'up_len':-retreat})
            rdf['w21t_glacier_number'] = selrow.w21t_glacier_number
        rdfs.append(rdf)

    rdf = pd.concat(rdfs)
    return rdf
# -------------------------------------------------------------------
import collections
import itertools

# https://codereview.stackexchange.com/questions/82010/averaging-lists-of-values-with-duplicate-keys
def avg_dups_fast(genes, values):
    # Find the sorted indices of all genes so that we can group them together
    sorted_indices = np.argsort(genes)
    # Now create two arrays using `sorted_indices` where similar genes and
    # the corresponding values are now grouped together
    sorted_genes = genes[sorted_indices]
    sorted_values = values[sorted_indices]
    # Now to find each individual group we need to find the index where the
    # gene value changes. We can use `numpy.where` with `numpy.diff` for this.
    # But as numpy.diff won't work with string, so we need to generate
    # some unique integers for genes, for that we can use
    # collections.defaultdict with itertools.count. 
    # This dict will generate a new integer as soon as a
    # new string is encountered and will save it as well so that same
    # value is used for repeated strings. 
    d = collections.defaultdict(itertools.count(0))
    unique_ints = np.fromiter((d[x] for x in sorted_genes), dtype=int)
    # Now get the indices
    split_at = np.where(np.diff(unique_ints)!=0)[0] + 1
    # split the `sorted_values` at those indices.
    split_items = np.array_split(sorted_values, split_at)
    return np.unique(sorted_genes), np.array([np.mean(arr, axis=0) for arr in split_items])

def avg_dups_python(genes, values):
    d = collections.defaultdict(list)
    for k, v in zip(genes, values):
        d[k].append(v)
#    return np.arraylist(d), [np.mean(val, axis=0) for val in d.values()]    
    return np.array(list(d)), np.array([np.mean(val, axis=0) for val in d.values()])
# -------------------------------------------------------------------



def timeseries_df(select):
    """Reads the per-glacier Wood data files to retrieve linear retreat.
    Returns dataframe:
        year:
            Time of measurement (can be franctional year)
        up_len:
            Number gets smaller as glacier retreats; can be negative.
            This column is de-meaned
        w21t_glacier_number:
    """
    # output vname, group, vname_time, vname_val
    specs = (
        ('retreat_fn', ('ice_front_retreat', 'discrete'), 'retreat_time', 'retreat'),
        ('qsg_fn', ('subglacial_discharge',), 'subglacial_discharge_time', 'subglacial_discharge'),
        ('tf_fn', ('thermal_forcing','terminus'), 'model_time', 'model_thermal_forcing'),
    )

    rows = list()
    for _,selrow in select.df.iterrows():
        row = {'w21t_glacier_number': selrow.w21t_glacier_number}
        with netCDF4.Dataset(os.path.join('data', 'wood2021', 'data', selrow['w21_data_fname'])) as nc:
            for ovname, sgroups, vname_time, vname_val in specs:
                # Go down path of groups
                grp = nc
                for sgroup in sgroups:
                    grp = grp[sgroup]

                time = grp.variables[vname_time][:].astype(np.float64)
                val = grp.variables[vname_val][:].astype(np.float64)
                time,val = avg_dups_python(time, val)

                val_fn = scipy.interpolate.interp1d(time, val, kind='cubic', fill_value=-1.e10)
                row[ovname] = (val_fn, time[0], time[-1])

        rows.append(row)

    return pd.DataFrame(rows)

# -------------------------------------------------------------------

def join_retreats_q4tf(rdf,wdfs,decade_mean=True):
    """
    rdf:
        Result of read_retreats()
    wdfs:
        Result of wood_q4tf()
    mean:
        True: Average over points to give just one per decade in Wood data
        False: Give individual points

    Returns mdfs[] (rdf joined with each in wdfs):
        w21t_Glacier
        w21t_glacier_number
        year0
        year1
        year
        up_len
        w21_mean_fjord_width
        Qsg
        TF
        q4tf
    """

    mdfs = list()

    # Join rdf (year,up_len) to appropriate row of q4tf
    # https://pandas.pydata.org/pandas-docs/version/0.20/generated/pandas.merge_asof.html#pandas.merge_asof
    for wdf in wdfs:
        wdf = wdf.sort_values('year0')
        rdf = rdf.sort_values('year')
        mdf = pd.merge_asof(rdf,wdf,left_on='year',right_on='year0', by=['w21t_glacier_number'])
        mdf = mdf.dropna()
        
        # Get just 1 point per decade
        if decade_mean:
            dfg = mdf.groupby(['w21t_Glacier', 'w21t_glacier_number','year0','year1'])
            mdf = dfg.mean()
        mdf = mdf.reset_index()

        mdfs.append(mdf)

    return mdfs

# ---------------------------------------------------------------
def regress_kappas(mdfs):
    """Determins kappa for up_len ~ q4tf (for the two regional glacier subsets"""

    kappas_list = list()
    for mdf in mdfs:

        dfg = mdf.groupby(['w21t_Glacier','w21t_glacier_number'])
        kappas = list()
        for _,df in dfg:
            regr = statsmodels.formula.api.ols('up_len ~ q4tf', data=df).fit()
            #print(regr.summary())
            kappa = regr.params.q4tf   # Kappa named after regression in paper
            kappas.append(kappa)
        ##    #lr = sklearn.liear_model.LinearRegression()
         #   lr.fit()
        #    print(df)
        #    df.plot.scatter('q4tf','up_len')
        kappas = np.array(kappas)
        df = pd.DataFrame({'kappa':kappas})
        df = df[df.kappa.abs()<1]
        df = df.sort_values('kappa').reset_index()
        df['kappa'].plot()
        print(df.kappa.mean())
        kappa = df.kappa.mean()   # delta_L = kappa * delta_q4tf
        kappas_list.append(kappa)

    return kappas_list


