import pandas as pd
import numpy as np
import seaborn as sns
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy import spatial
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, precision_score, recall_score,
    recall_score, roc_curve, RocCurveDisplay, auc
)
from tqdm import tqdm, trange
from scipy import stats, spatial
from sklearn.cluster import DBSCAN


def is_outlier_tukey(vector: np.ndarray, k: float = 1.5) -> np.ndarray:
    """
    Find outliers in a numeric vector using Tukey's criteria

    Args:
        vector: Numeric vector to analyze
        k: Threshold to be consider a outlier in IQRs

    Returns:
        (bool): Boolean vector O, where o_i is True if vector_i is a outlier, False otherwise
    """
    p_25_75 = np.percentile(vector, [25, 75])
    iqr = p_25_75[1] - p_25_75[0]
    return (vector < (p_25_75[0] - k * iqr)) | (vector > (p_25_75[1] + k * iqr))


tqdm.pandas()


PERIODS = [7, 15, 30, 180, 365]  # periods to search (apart from "1")
C = 2 * np.pi * 1  # unit circunference length
one_deg_arc_length = C / 360
one_day_arc_length = {p: C / p for p in PERIODS}


def percentage_distance(x,y):
    # |x-y| / min(x, y)
    return np.abs(y-x) / (np.minimum(np.abs(x), np.abs(y)) + 1e-5)


def _diff_bin(array, thresh=.3):
    """ Bucketizes 1D arrays by splitting the values where their percent difference is greater than `thresh`.
    Returns array with "Cluster IDs" for the array
    """
    uniques = np.sort(np.abs(np.unique(array)))
    array_sorted = np.sort(uniques)
    # safe divide, 0 on NaN
    percent_diffs = np.divide(np.diff(array_sorted), array_sorted[1:], out=np.zeros_like(array_sorted[1:]), where=array_sorted[1:]!=0)
    markers = np.abs(percent_diffs) > thresh
    return np.digitize(np.abs(array), array_sorted[1:][markers])


def _clust_am(data, min_samples=2, eps=.2):
    """ This fn is intended to work on an groupby-apply basis.
    Takes a "dataframe" with an "amount" column  and runs 1-D DBSCAN on it.
    Returns the cluster IDs for each sample and a second column with the original
    group orders (for later joining to the original df).
    """
    x = data.amount.values.reshape(-1,1)
    orders = data.group_position
    
    y = _diff_bin(x, thresh=eps)

    # Flag as noise values that appear too few times
    uniques, counts = np.unique(y, return_counts=True)
    invalid_vals = uniques[np.argwhere(counts < min_samples).flatten()]
    for invalid_val in invalid_vals:
        y[y == invalid_val] = -1
    y = y.flatten()
    
    return pd.Series({'amount_lvl': y, 'group_position': orders.values})


def circunference_distance(x, y, unit_arc_len=one_deg_arc_length):
    return np.degrees(
        np.arccos(np.clip(np.dot(x, y), -1.0, 1.0))
    ) * one_deg_arc_length / unit_arc_len



def detect_temporal_recurrencies(datediffs, eps_mult=.1, min_samples=3): 
    data = datediffs
    is_daily = False

    if len(data) < 3:   # Only check if enough events exist
        ret = np.ones(len(data)) * -1
    else:
        datediffs_to_previous = datediffs#np.diff(data, 1) #pd.to_datetime(data['date']).diff(1).dt.days.fillna(0)
        outliers = is_outlier_tukey(datediffs_to_previous[1:])
        p = np.median(datediffs_to_previous[1:][~outliers]) # From 1 'cause the first one will be 0
        
        if p < 2: # daily recurrence case: if most deltas between events are "1".
            ret = np.ones(len(data))
        
        else:  # recurrence could still match at certain period
            
            # check, from our PERIODS, which is the one closest to the 
            # datediffs median (i.e. p)
            best_matching_period = PERIODS[np.argmin(
                [np.abs(period - p) for period in PERIODS]
            )]
            
            # distances to the beginning of the serie 
            datediffs = datediffs_to_previous.cumsum()
            
            # Mods of the distances
            mods = datediffs % best_matching_period
            
            # Because distances at the beginning and end of the ranges are "close"
            # between each other, we need to encode them as cyclic features (sin, cos).
            # Example with PERIOD=30:
            # (distance % 30 = 1) ≈ (distance % 30 = 29)
            
            # Project distances on the unit circle
            mods_cos = np.cos(2*np.pi*mods/best_matching_period)
            mods_sin = np.sin(2*np.pi*mods/best_matching_period)
            X = np.vstack((mods_cos, mods_sin)).T
            
            # Support dynamic threshold (depending on PERIOD, allowed period should be greater)
            # (chord of one day)
            eps =  best_matching_period * eps_mult  #max(distance_1day, distance_1day * ( best_matching_period * eps_mult ))   # TODO: disabled
            
            
            mod_clusters = DBSCAN(metric=circunference_distance, 
                                  min_samples=min_samples, 
                                  eps=eps,
                                  algorithm='auto',
                                  metric_params={'unit_arc_len': one_day_arc_length[best_matching_period]}).fit_predict(X)
            ret = mod_clusters.flatten()

            #t = .3
            #clusters_x = _diff_bin(mods_cos, thresh=t)
            #clusters_y = _diff_bin(mods_sin, thresh=t)
            #ret = 10 * clusters_x + clusters_y
            ## Flag as noise values that appear too few times
            #uniques, counts = np.unique(ret, return_counts=True)
            #invalid_vals = uniques[np.argwhere(counts < 2).flatten()]
            #for invalid_val in invalid_vals:
            #    ret[ret == invalid_val] = -1
            #ret = ret.flatten()

            
            # All daily recurrencies (1,2) cannot be part of a cluster with PERIOD > 1
            # DBSCAN will asign the same cluster to period=1 and to period=30 (for example)
            ret[1:][datediffs_to_previous[1:] < eps] = -1
            
            #print(mods)
            
    return ret

def detect_recurrencies_wrapped(data, eps_mult=.1, min_samples=3):
    """ Wrap the detect recurrencies algo so the result can be merged back with 
    the original df
    """
    diffs = pd.to_datetime(data.date).diff(1).dt.days.fillna(0)
    ret = detect_temporal_recurrencies(diffs, eps_mult=eps_mult, min_samples=min_samples)
    group_positions = data.group_position.values
    return pd.Series({'day_cluster': ret, 'group_position': group_positions})


def detect_breaks(data, threshold=.1):
    """ Marks when a serie "breaks". That is, it stops happening for a while.
    Marks as "True" the events whose distance to its previous is greater than the 
    median difference * (1 + threshold).
    """
    datediffs_to_previous = data.diffdays #pd.to_datetime(data['date']).diff(1).dt.days.fillna(0)
    median_period = datediffs_to_previous.median()
    threshold = median_period * (1 + threshold)
    is_greater = (data.diffdays > threshold) | (data.diffdays == 0)
    return is_greater.values


def detect_breaks_wrapped(data, threshold=.1):
    """ Wrap the detect breaks algo so the result can be merged back with 
    the original df
    """
    ret = detect_breaks(data, threshold=threshold)
    group_positions = data.group_position.values
    return pd.Series({'is_break': ret, 'group_position': group_positions})


def apply_clust_amount(df, pk='payment_channel', sort_by='date', eps=.1):
    bin_nbs = df.sort_values(['payment_channel', 'date'])\
        .groupby(['payment_channel'])\
        .apply(_clust_am, min_samples=4, eps=eps)\
        .explode(['amount_lvl', 'group_position'])

    df = pd.merge(df, bin_nbs, on=['payment_channel', 'group_position'])
    return df

def apply_detect_temporal_recurrencies(df, eps=.1):
    df['is_rec'] = 0
    lelele = df.sort_values(['payment_channel', 'amount_lvl', 'date'])\
        .groupby(['payment_channel', 'amount_lvl'])\
        .progress_apply(detect_recurrencies_wrapped, eps_mult=eps, min_samples=4)    # Was .1

    lelele = lelele.explode(['day_cluster', 'group_position'])
    df = pd.merge(df, lelele, on=['payment_channel', 'amount_lvl', 'group_position'])
    df.loc[:, 'is_rec'] = 1
    df.loc[(df.amount_lvl == -1) | (df.day_cluster == -1), 'is_rec'] = 0
    return df

def main_dbscan_method(df, eps_amount=.1, eps_date=.1):
    df = apply_clust_amount(df, eps=eps_amount)
    df = apply_detect_temporal_recurrencies(df, eps=eps_date)
    return df
