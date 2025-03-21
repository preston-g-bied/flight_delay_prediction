# src/features/interaction_features.py

import pandas as pd
import numpy as np

def create_interaction_features(df):
    """
    Create interaction features based on EDA insights.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with existing features
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with interaction features added
    """
    df_out = df.copy()
    
    # Hub-to-hub flights
    if 'origin_is_hub' in df_out.columns and 'dest_is_hub' in df_out.columns:
        df_out['hub_to_hub'] = df_out['origin_is_hub'] * df_out['dest_is_hub']
    
    # Peak travel season on weekend
    if 'is_weekend' in df_out.columns and 'is_peak_travel_season' in df_out.columns:
        df_out['peak_weekend'] = df_out['is_weekend'] * df_out['is_peak_travel_season']
    
    # Evening flights on weekend
    if 'time_period' in df_out.columns and 'is_weekend' in df_out.columns:
        df_out['evening_weekend'] = ((df_out['time_period'] == 'Evening') | 
                                    (df_out['time_period'] == 'Night')).astype(int) * df_out['is_weekend']
    
    # Major carrier at hub airport
    if 'carrier_size_rank' in df_out.columns and 'origin_is_hub' in df_out.columns:
        df_out['major_carrier_at_hub'] = (df_out['carrier_size_rank'] > 0.8).astype(int) * df_out['origin_is_hub']
    
    # Long distance during peak travel season
    if 'distance_category' in df_out.columns and 'is_peak_travel_season' in df_out.columns:
        df_out['long_distance_peak'] = ((df_out['distance_category'] == 'Long') | 
                                      (df_out['distance_category'] == 'Very Long')).astype(int) * df_out['is_peak_travel_season']
    
    # High risk combinations: problematic carrier on problematic route
    if 'carrier_delay_rate' in df_out.columns and 'route_delay_rate' in df_out.columns:
        carrier_mean = df_out['carrier_delay_rate'].mean()
        route_mean = df_out['route_delay_rate'].mean()
        df_out['high_risk_combo'] = ((df_out['carrier_delay_rate'] > carrier_mean) & 
                                    (df_out['route_delay_rate'] > route_mean)).astype(int)
    
    # Morning rush hour (6-9 AM)
    if 'dep_hour' in df_out.columns:
        df_out['morning_rush'] = df_out['dep_hour'].between(6, 9).astype(int)
        
        # Evening rush hour (4-7 PM)
        df_out['evening_rush'] = df_out['dep_hour'].between(16, 19).astype(int)
        
        # Rush hour at busy airport
        if 'origin_freq_rank' in df_out.columns:
            df_out['rush_at_busy_airport'] = (df_out['morning_rush'] | df_out['evening_rush']).astype(int) * (df_out['origin_freq_rank'] > 0.8).astype(int)
    
    # Weather risk: winter months in northern hubs
    if 'Month' in df_out.columns and 'Origin' in df_out.columns:
        # Define northern hubs (this is a simplified approach)
        northern_hubs = ['ORD', 'DTW', 'MSP', 'BOS', 'JFK', 'LGA', 'EWR', 'CLE', 'PIT', 'SEA']
        winter_months = [11, 12, 1, 2, 3]  # Nov-Mar
        
        df_out['winter_in_north'] = ((df_out['Month'].isin(winter_months)) & 
                                    (df_out['Origin'].isin(northern_hubs))).astype(int)
    
    # Departure delay risk score - combining multiple risk factors
    risk_factors = []
    for col in ['high_risk_combo', 'rush_at_busy_airport', 'winter_in_north', 'peak_weekend']:
        if col in df_out.columns:
            risk_factors.append(col)
    
    if risk_factors:
        df_out['delay_risk_score'] = df_out[risk_factors].sum(axis=1)
    
    return df_out