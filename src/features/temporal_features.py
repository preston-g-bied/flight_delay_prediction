# src/features/temporal_features.py

import pandas as pd
import numpy as np
from datetime import datetime

def create_temporal_features(df):
    """
    Create time-based features from the flight data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with the raw flight data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with temporal features added
    """
    # create a copy to avoid modifying the original
    df_out = df.copy()

    # extract the hour and minute from departure time
    df_out['dep_hour'] = df_out['DepTime'] // 100
    df_out['dep_minute'] = df_out['DepTime'] % 100

    # time of day categories
    df_out['time_period'] = pd.cut(
        df_out['dep_hour'],
        bins = [0, 5, 11, 17, 23],
        labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    )

    # weekend indicator
    df_out['is_weekend'] = (df_out['DayOfWeek'] >= 6).astype(int)

    # season from month
    seasons = {
        1: 'Winter', 2: 'Winter', 3: 'Spring', 
        4: 'Spring', 5: 'Spring', 6: 'Summer',
        7: 'Summer', 8: 'Summer', 9: 'Fall', 
        10: 'Fall', 11: 'Fall', 12: 'Winter'
    }
    df_out['season'] = df_out['Month'].map(seasons)

    # create a month-day string for holiday detection
    df_out['month_day'] = df_out['Month'].astype(str).str.zfill(2) + '-' + df_out['DayofMonth'].astype(str).str.zfill(2)

    # US Holiday indicators (simplified)
    us_holidays = [
        '01-01',  # New Year's Day
        '01-15',  # Martin Luther King Jr. Day (third Monday in January, approximated)
        '02-14',  # Valentine's Day
        '02-15',  # Presidents' Day (third Monday in February, approximated)
        '03-17',  # St. Patrick's Day

        # Easter varies by year but falls between March 22 and April 25
        # I'll include common dates in this range
        '04-05',  # Easter approximation (early April)
        '04-15',  # Easter approximation (mid April)
    
        '05-05',  # Cinco de Mayo
        '05-25',  # Memorial Day (last Monday in May, approximated)
        '06-19',  # Juneteenth
        '07-04',  # Independence Day
        '09-01',  # Labor Day (first Monday in September, approximated)
        '10-31',  # Halloween
        '11-11',  # Veterans Day
        '11-25',  # Thanksgiving (fourth Thursday in November, approximated)
        '11-26',  # Black Friday
        '12-24',  # Christmas Eve
        '12-25',  # Christmas Day
        '12-31',  # New Year's Eve
    ]
    df_out['is_holiday'] = df_out['month_day'].isin(us_holidays).astype(int)

    # peak travel seasons
    summer_peak = (df_out['Month'] >= 6) & (df_out['Month'] <= 8)
    winter_holiday = ((df_out['Month'] == 12) & (df_out['DayofMonth'] >= 15)) | (df_out['Month'] == 1 & (df_out['DayofMonth'] <= 5))
    df_out['is_peak_travel_season'] = (summer_peak | winter_holiday).astype(int)
    
    return df_out