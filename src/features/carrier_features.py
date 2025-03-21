# src/features/carrier_features.py

import pandas as pd
import numpy as np

def create_carrier_features(df, train_data=None):
    """
    Create carrier-related features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with the raw flight data
    train_data : pandas.DataFrame, optional
        Training data to extract carrier statistics (for test data transformation)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with carrier features added
    """
    # use the provided dataframe or itself as reference data
    reference_data = train_data if train_data is not None else df
    df_out = df.copy()
    
    # carrier size and frequency
    carrier_counts = reference_data['UniqueCarrier'].value_counts()
    carrier_rank = carrier_counts.rank(pct=True)
    
    df_out['carrier_size_rank'] = df_out['UniqueCarrier'].map(carrier_rank)
    
    # carrier delay rates
    if 'dep_delayed_15min' in reference_data.columns:
        if reference_data['dep_delayed_15min'].dtype == object:
            # convert Y/N to 1/0 if not already done
            reference_delay = reference_data.copy()
            reference_delay['dep_delayed_15min'] = reference_delay['dep_delayed_15min'].map({'Y': 1, 'N': 0})
        else:
            reference_delay = reference_data
            
        carrier_delay_rates = reference_delay.groupby('UniqueCarrier')['dep_delayed_15min'].mean()
        df_out['carrier_delay_rate'] = df_out['UniqueCarrier'].map(carrier_delay_rates)
    
    # create carrier-specific time features
    if 'dep_hour' in df_out.columns:
        # calculate carrier performance by time of day
        carrier_hour_delay = reference_delay.groupby(['UniqueCarrier', 'dep_hour'])['dep_delayed_15min'].mean().reset_index()
        carrier_hour_map = dict(zip(zip(carrier_hour_delay['UniqueCarrier'], carrier_hour_delay['dep_hour']), 
                                   carrier_hour_delay['dep_delayed_15min']))
        
        df_out['carrier_hour_performance'] = df_out.apply(
            lambda x: carrier_hour_map.get((x['UniqueCarrier'], x['dep_hour']), np.nan), axis=1
        )
    
    # create carrier-route performance features
    carrier_route_grouped = reference_data.groupby(['UniqueCarrier', 'Origin', 'Dest']).size().reset_index(name='route_carrier_count')
    
    # merge this information back (this is an alternative to map for multi-key lookups)
    df_out = pd.merge(
        df_out, 
        carrier_route_grouped,
        on=['UniqueCarrier', 'Origin', 'Dest'],
        how='left'
    )
    
    # fill missing values for routes not in training data
    df_out['route_carrier_count'] = df_out['route_carrier_count'].fillna(0)
    
    return df_out