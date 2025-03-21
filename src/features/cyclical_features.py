# src/features/cyclical_features.py

import numpy as np
import pandas as pd

def create_cyclical_features(df):
    """
    Create cyclical transformations of temporal features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with the raw flight data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with cyclical features added
    """
    df_out = df.copy()
    
    # Create hour of day cyclical features
    if 'dep_hour' in df_out.columns:
        df_out['dep_hour_sin'] = np.sin(2 * np.pi * df_out['dep_hour'] / 24)
        df_out['dep_hour_cos'] = np.cos(2 * np.pi * df_out['dep_hour'] / 24)
    elif 'DepTime' in df_out.columns:
        # Create dep_hour if it doesn't exist
        df_out['dep_hour'] = df_out['DepTime'] // 100
        df_out['dep_hour_sin'] = np.sin(2 * np.pi * df_out['dep_hour'] / 24)
        df_out['dep_hour_cos'] = np.cos(2 * np.pi * df_out['dep_hour'] / 24)
    
    # Create day of week cyclical features
    if 'DayOfWeek' in df_out.columns:
        df_out['day_of_week_sin'] = np.sin(2 * np.pi * (df_out['DayOfWeek'] - 1) / 7)
        df_out['day_of_week_cos'] = np.cos(2 * np.pi * (df_out['DayOfWeek'] - 1) / 7)
    
    # Create month cyclical features
    if 'Month' in df_out.columns:
        df_out['month_sin'] = np.sin(2 * np.pi * (df_out['Month'] - 1) / 12)
        df_out['month_cos'] = np.cos(2 * np.pi * (df_out['Month'] - 1) / 12)
    
    # Create day of month cyclical features
    if 'DayofMonth' in df_out.columns:
        # Get max days for normalization (approximating to 31 for simplicity)
        max_days = 31
        df_out['day_of_month_sin'] = np.sin(2 * np.pi * (df_out['DayofMonth'] - 1) / max_days)
        df_out['day_of_month_cos'] = np.cos(2 * np.pi * (df_out['DayofMonth'] - 1) / max_days)
    
    # Create time of day features that combine hour and minute
    if 'DepTime' in df_out.columns:
        # Convert time like 1430 (2:30 PM) to fraction of day
        hours = df_out['DepTime'] // 100
        minutes = df_out['DepTime'] % 100
        time_of_day = (hours * 60 + minutes) / (24 * 60)  # Fraction of day
        
        df_out['time_of_day_sin'] = np.sin(2 * np.pi * time_of_day)
        df_out['time_of_day_cos'] = np.cos(2 * np.pi * time_of_day)
    
    return df_out