# Utility module for adding artificial noise to series of events noise

from datetime import date, timedelta
from plotly.subplots import make_subplots
from random import randint
from matplotlib import colors

import random
import math
import pandas as pd
import numpy as np
import datetime


def standard_dev(x): 
    """
    Helper for applying as lambda during groupbys
    """
    return np.std(x, ddof=1)

def date_pertubation(d, minimum=0, maximum=1, std=None):
    '''
    Method that perturbs the date of an event adding days between a maximum and a minumum already established.
    If they're not passed, the method will add a perturbation coming from a normal of mean 0 and std=std
    '''
    if std is None:
        day_delta = randint(minimum, maximum)
        return pd.to_datetime(date(d.year, d.month, d.day) + timedelta(days=day_delta))     
    else:
        day_delta = np.random.normal(loc=0, scale=std)
        return pd.to_datetime(date(d.year, d.month, d.day) + timedelta(days=day_delta))
        
    
def amt_perturbation(d, std = None, low = 0.4, high = 0.9, mean = None):
    '''
    Method that perturbs a point by adding to the point a movement of a value 
    generated by a normal distribution --> by default with mean 0
    '''
    # TODO: Sample from a distribution that is the inverse of a Gaussian in such a way
    # that sampling values close to 0 is rare (minimal perturbation)
    #amt_update = d + (np.random.normal(mean, std))
    
    # TODO: Add exponential to add a smaller delta as the median increases
    # (it's not the same 10% of t=year as of t=week)
    
    if mean is None:
        amt_update = d + d * np.random.uniform(low=low, high=high) 
            
    else:
        amt_update = d + (np.random.normal(mean, std))
    
    return amt_update


def detect_breaks(data, threshold=.1):
    """ Marks when a serie "breaks". That is, it stops happening for a while.
    Marks as "True" the events whose distance to its previous is greater than the 
    median difference * (1 + threshold).
    """
    datediffs_to_previous = data.date_diffdays #pd.to_datetime(data['date']).diff(1).dt.days.fillna(0)
    median_period = datediffs_to_previous.median()
    threshold = median_period * (1 + threshold)
    is_greater = (data.date_diffdays > threshold) | (data.date_diffdays == 0)
    return is_greater.values

def detect_breaks_wrapped(data, **kwargs):
    """ Wrap the detect breaks algo so the result can be merged back with 
    the original df
    """
    ret = detect_breaks(data, **kwargs)
    group_positions = data.group_position.values
    return pd.Series({'is_break': ret, 'group_position': group_positions})

def add_noise_update_exp1(dataframe, 
                          n_desv_outlier=3, 
                          prob_perturbation=0.05,
                          col_pk = 'payment_channel',
                          col_amnt = 'amount',
                          col_date = 'date',
                         ):
    """
    EXPERIMENT 1: ADD NOISE WITHOUT MOVING EXISTING POINTS
    """
    dataframe = dataframe.sort_values(col_date)
    
    # Keep old amounts and dates
    dataframe['amount_old'] = dataframe[col_amnt].values
    dataframe['date_old'] = dataframe[col_date].values
    
    # Flag original movs as recurrent
    dataframe['is_rec']=1
    
    # Compute day differences between series
    lelele = dataframe.sort_values([col_pk, col_date])\
        .groupby(col_pk)[col_date]\
        .diff(1).dt.days

    if col_date+'_diffdays' not in dataframe.columns:
        dataframe = dataframe.join(lelele, rsuffix='_diffdays')


    # compute series stats
    mean_std = dataframe.groupby(col_pk).agg({col_amnt:['mean', standard_dev], col_date:['min','max'], col_date+'_diffdays':(lambda x: np.nanmedian(x[x>0]))})
    mean_std.columns = mean_std.columns.droplevel(0)
    mean_std.columns = ['mean', 'standard_dev', 'min', 'max', 'median']
    dataframe = dataframe.join(mean_std, on=col_pk, how='left')

    median_diff = np.nanmedian(dataframe.date_diffdays.values)
    dataframe['median'] = dataframe['median'].fillna(median_diff) # When there's only one mov, the median is itself
    
    df1 = dataframe.copy()

    """
    1. Strict Date Method:
        - Perturbing dates within the interval [0.3 median period, 0.6 median period] 
        to ensure that the new noise points are not too close to existing points 
        and that the algorithm does not confuse them.
        - Perturbing amounts with a normal distribution of parameters (mean of the amounts, 
        3 x standard deviation), where the mean and standard deviation values 
        are calculated using the distribution of amounts in that series.
    """    

    # Create 2 idx 
    df1['new_point_exp_11_ind'] = np.where(np.random.random((len(df1),1))<prob_perturbation,1,0) # indicador experimento 1.1
    df1['new_point_exp_12'] =  df1['new_point_exp_11_ind'].apply(lambda x: np.random.random() if x == 0 else 1)
    df1['new_point_exp_12_ind'] = np.where(df1['new_point_exp_12']<prob_perturbation,1,0)
    df1.drop('new_point_exp_12', axis=1, inplace=True)
        
    # Update:
    # - DATES in an interval (default [0.3, 0.6] times median) 
    # - AMOUNTS with std * n. --> default 3 and mean of amounts
        

    # Make a subset and update columns
    subset_df11 = df1.loc[(df1['new_point_exp_11_ind']==1)]

    subset_df11[col_date] = subset_df11.apply(lambda x: date_pertubation(x[col_date], np.floor(0.3*x['median']), np.ceil(0.6*x['median'])), axis=1) 
    subset_df11[col_amnt] = subset_df11.apply(lambda x: amt_perturbation(x[col_amnt], 
                                                                         std=x['standard_dev'], 
                                                                         mean=None), axis=1) 

    
    # Añadimos estos casos al df - estos casos se cambian como ruido ya que no son recurrentes
    subset_df11['is_rec'] = 0
    df1 = pd.concat([df1, subset_df11])
    df1 = df1.reset_index(drop=True)
        
    """
    2. Strict Amounts Method:
        - Perturbing DATES within the interval (0 median period, median period)
        being more lenient with dates.
        - Perturbing the amount more significantly with a normal distribution of parameters
        (mean of the amounts, n x standard deviation --> n default 3).
        We add a condition that the simulated amount is greater in absolute value
        than twice the mean; otherwise, we set it to twice the mean.

        Note: with the Gaussian centered on the median, the most probable points will still be 
        those of the median, and there is no guarantee that we are perturbing the amount 
        significantly in all cases. Therefore, we must demand more if we want to ensure 
        perturbation of amounts when we are more lenient with dates.
    """

    
    # Make a subset for the actual points and update columns
    subset_df12 = df1.loc[(df1['new_point_exp_12_ind']==1)]
    subset_df12['aleatorio'] = np.random.uniform(0, subset_df12['median'], len(subset_df12))      

    subset_df12[col_date] = subset_df12.apply(lambda x: date_pertubation(x[col_date], 0, np.ceil(x['median'])), axis=1)  # was 1, round()
    subset_df12[col_amnt] = subset_df12.apply(lambda x:  np.sign(x[col_amnt]) * np.maximum(    # maintain sign of the original mov
                                                                np.abs(x['median'] * 2), 
                                                                np.abs(amt_perturbation(x[col_amnt], std=max(x['standard_dev'], x['median']*.1) * n_desv_outlier, mean=x['mean']))), 
                                               axis=1)


    # Add these cases. These are swapped as noise as they won't be recurrent.
    subset_df12['is_rec']=0
    
    df1 = pd.concat([df1, subset_df12])
    return df1


def add_noise_update_exp2(dataframe, 
                          prob_combination=0.05,
                          col_pk = 'payment_channel',
                          col_amnt = 'amount',
                          col_date = 'date',
                          noise_type = 'combine'
                          ):
    
    '''
    EXPERIMENT 2: Overlapping Series. In this experiment, we combine
    two series into one to observe the algorithm's capability in separating series,
    without adding extra noise.
    
    Parameters:
    - dataframe: transactions pandas DataFrame
    - prob_combination: percentage of primary keys (PKs) to combine
    - col_pk: group column (Payment channel)
    - col_amnt: amount column
    - col_date: date column
    - noise_type: Type of noise to include -->
                    - combine: combine two series as one
                    - offset: introduce an offset in amount or date
    - offset_type: if noise type is offset, this specifies whether it's offsetting the amount or the date
    - prob_offset: percentage increment of the offset applied to the period of dates or the current amount

    '''
   
    # Get order by date within payment channels. 
    # This will help on later calculations
    dataframe = dataframe.sort_values(['payment_channel', 'date'])
    posid = dataframe.groupby(['payment_channel'])\
                .cumcount().rename('group_position')
    dataframe = dataframe.join(posid)
    
    ## Compute the difference between days
    lelele = dataframe.sort_values([col_pk, col_date])\
        .groupby(col_pk)[col_date]\
        .diff(1).dt.days

    if col_date+'_diffdays' not in dataframe.columns:
        dataframe = dataframe.join(lelele, rsuffix='_diffdays')


    df1 = dataframe

    
    #Add stats
    mean_std = df1.groupby(col_pk).agg({col_amnt:['mean', standard_dev], col_date:['min','max'], col_date+'_diffdays':(lambda x: np.nanmedian(x[x>0]))})
    mean_std.columns = mean_std.columns.droplevel(0)
    mean_std.columns = ['mean', 'standard_dev', 'min', 'max', 'median']
    mean_std['rndm_date'] = np.random.uniform(low=0.1, high = 0.5, size=len(mean_std))
    mean_std['rndm_amnt'] = np.random.uniform(low=0.3, high = 0.6, size=len(mean_std))
    
    df1 = df1.join(mean_std, on=col_pk, how='left')

    median_diff = np.nanmedian(df1.date_diffdays.values)
    df1['median'] = df1['median'].fillna(median_diff) # When serie has only one mov, it doesn't have median.
    
    
    # Merge random pairs of series into ones
    # Create the flag for doing it
    unique_series = df1[col_pk].unique()
    ind_merge = np.random.random(len(unique_series)) < prob_combination
    unique_series = pd.DataFrame({col_pk: unique_series, 'comb_serie_exp_2_ind': ind_merge})

    
    df1 = pd.merge(df1, unique_series, on=col_pk)
    print(df1.shape[0])
    pks_comb = df1[df1['comb_serie_exp_2_ind']==1].payment_channel.unique() # PKs of series we'll merge
    if noise_type=='combine':
        pks_comb = [pks_comb[i: i + len(pks_comb)//2] for i in range(0, len(pks_comb), len(pks_comb)//2) ] # create two sublists
        
        pks_comb_dict = dict(zip(pks_comb[0], pks_comb[1])) #  Transform them to <key: value> format


        replace = df1[df1['payment_channel'].isin(set(pks_comb_dict.keys()))]
        not_replace = df1[~df1['payment_channel'].isin(set(pks_comb_dict.keys()))]
        
        # Add original payment channel when combined for evaluation
        replace['payment_channel_ori'] = replace['payment_channel']
        not_replace['payment_channel_ori'] = not_replace['payment_channel']
        
        replace[col_pk] = replace[col_pk].map(pks_comb_dict) 
    
    else:
        not_replace = df1.copy()
        not_replace['offset_introduced'] = 0
        replace = df1[df1['payment_channel'].isin(pks_comb)]
        replace['offset_introduced'] = 1
        replace[col_amnt] = replace[col_amnt]*(1+replace['rndm_amnt'])        
        replace[col_date] = replace[col_date] + pd.to_timedelta(np.ceil(replace['rndm_date'] * replace['median']), unit='day')
        
    df1 = pd.concat([replace, not_replace])
    df1['is_rec'] = 1
                            
    return df1


def add_noise_update_exp3(dataframe, 
                          perc_update=0.15, 
                          perc_deviate=0.05, 
                          perc_comb=0.05, 
                          n_desv_outlier=.15, 
                          prob_perturbation=0.05,
                          col_pk = 'payment_channel',
                          col_amnt = 'amount',
                          col_date = 'date',
                          run_noise_1 = False,
                          n_desv_outlier_exp1=3,
                         ):

    """Experiment 3: We add noise by perturbing existing points and adding new noise.

    - New noise is added following the rules for dates and amounts from Experiment 1. (can be deactivated)
    - Perturbation to create noise (existing point that will be considered an outlier): following the same rules as Experiment 1.
    - Perturbation to move points (which will NOT be considered outliers): the existing amount value is assigned the value of a normal with 
                                                                          mean 0 and the deviation of the amount in that PK, and random values are added to the dates ~ N(0, 0.15 x median).
    """
    dataframe = dataframe.sort_values([col_pk, col_date])
    
    
    dataframe['is_rec']=1

    # Calculate the difference in days between transfers
    lelele = dataframe.sort_values([col_pk, col_date])\
        .groupby(col_pk)[col_date]\
        .diff(1).dt.days
    dataframe = dataframe.join(lelele, rsuffix='_diffdays')

    # Calculate statistics for each series
    #mean_std = dataframe.groupby(col_pk).agg({col_amnt:['mean', standard_dev], col_date:['min','max'], col_date+'_diffdays':['median']})
    nan_median = lambda x: np.nanmedian(x.values[x.values>0])
    nan_median.__name__ = 'median'
    mean_std = dataframe.groupby(col_pk).agg({col_amnt:['mean', standard_dev], col_date:['min','max'], col_date+'_diffdays': nan_median })
    mean_std.columns = mean_std.columns.droplevel(0)
    mean_std.columns = ['mean', 'standard_dev', 'min', 'max', 'median']

    # Keep old amounts and dates for checks (can be removed later)
    dataframe['amount_old'] = dataframe[col_amnt]
    dataframe['date_old'] = dataframe[col_date]
    

    noised_df = dataframe
                          
                             
    if run_noise_1:               
        noised_df = add_noise_update_exp1(noised_df, 
                                          n_desv_outlier=3,
                                          prob_perturbation=prob_perturbation,
                                          col_pk =col_pk,
                                          col_amnt = col_amnt,
                                          col_date = col_date)
    



        for c in ['mean', 'median', 'min', 'max', 'standard_dev']:
            if c in noised_df.columns:
                noised_df = noised_df.drop(c, axis=1)

    noised_df = noised_df.join(mean_std, on=col_pk, how='left')
    median_diff = np.nanmedian(noised_df.date_diffdays.values)
    noised_df['median'] = noised_df['median'].fillna(median_diff) # for when the serie has only one movement
    
    df1 = noised_df
    
    df1['new_point_exp_32_ind'] = np.where(np.random.random((len(df1),1))<prob_perturbation,1,0) # indicador experimento 3.2
    df1['new_point_exp_32'] = df1['new_point_exp_32_ind'].apply(lambda x: np.random.random() if x == 0 else 1)
    df1['new_point_exp_33_ind'] = np.where(df1['new_point_exp_32']<prob_perturbation,1,0)
    df1.drop('new_point_exp_32', axis=1, inplace=True)
    
   
    #######################
    ## Case Experiment 3.2
    df1[col_date] = df1.apply(lambda x: date_pertubation(x[col_date],round(0.3*x['median']), round(0.6*x['median'])) if x['new_point_exp_32_ind'] == 1 else x[col_date], axis=1) 
    df1[col_amnt] = df1.apply(lambda x: amt_perturbation(x[col_amnt], std=x['standard_dev'] * 3, mean=None) if x['new_point_exp_32_ind'] == 1 else x[col_amnt], axis=1)
    df1.loc[df1['new_point_exp_32_ind'] == 1, 'is_rec'] = 0
                             
    
    #######################
    ## Case Experiment 3.3    
    df1[col_amnt] = df1.apply(lambda x:  amt_perturbation(x[col_amnt], mean= 0, std=x['standard_dev']) if x['new_point_exp_33_ind'] == 1 else x[col_amnt], axis=1)
    df1[col_date] = df1.apply(lambda x: date_pertubation(x[col_date], std=n_desv_outlier * x['median']) if x['new_point_exp_33_ind'] == 1 else x[col_date], axis=1)
    df1.loc[df1['new_point_exp_33_ind'] == 1, 'is_rec'] = 1  # these are not outliers

    return df1